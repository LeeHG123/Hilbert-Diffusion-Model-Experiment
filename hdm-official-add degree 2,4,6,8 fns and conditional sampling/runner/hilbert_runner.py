import os
import logging
from scipy.spatial import distance
import numpy as np
import time
import tqdm
from tqdm import tqdm

from evaluate.power import calculate_ci
from datasets import data_scaler, data_inverse_scaler

from collections import OrderedDict

import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler

from models import *

from functions.utils import *
from functions.loss import hilbert_loss_fn
from functions.sde import VPSDE1D
from functions.sampler import sampler

torch.autograd.set_detect_anomaly(True)

def kernel_se(x1, x2, hyp={'gain':1.0,'len':1.0}):
    """ Squared-exponential kernel function """
    x1 = x1.cpu().numpy()
    x2 = x2.cpu().numpy()

    D = distance.cdist(x1/hyp['len'],x2/hyp['len'],'sqeuclidean')
    K = hyp['gain']*np.exp(-D)
    return torch.from_numpy(K).to(torch.float32)

class HilbertNoise:
    def __init__(self, grid, x=None, hyp_len=1.0, hyp_gain=1.0, use_truncation=False):
        x = torch.linspace(-10, 10, grid)
        self.hyp = {'gain': hyp_gain, 'len': hyp_len}
        x = torch.unsqueeze(x, dim=-1)
        self.x = x
        if x is not None:
            self.x=x

        K = kernel_se(x, x, self.hyp)
        K = K.cpu().numpy()
        eig_val, eig_vec = np.linalg.eigh(K + 1e-6 * np.eye(K.shape[0], K.shape[0]))

        self.eig_val = torch.from_numpy(eig_val)
        self.eig_vec = torch.from_numpy(eig_vec).to(torch.float32)
        self.D = torch.diag(self.eig_val).to(torch.float32)
        self.M = torch.matmul(self.eig_vec, torch.sqrt(self.D))

    def sample(self, size):
        size = list(size)  # batch*grid
        x_0 = torch.randn(size)

        output = (x_0 @ self.M.transpose(0, 1))  # batch grid x grid x grid
        return output  # bath*grid

    def free_sample(self, free_input):  # input (batch,grid)

        y = torch.randn(len(free_input), self.x.shape[0]) @ self.eig_vec.T @ kernel_se(self.x, free_input[0].unsqueeze(-1), self.hyp)
        return y

class HilbertDiffusion(object):
    def __init__(self, args, config, dataset, test_dataset, device=None):
        self.args = args
        self.config = config
        self.W = HilbertNoise(grid=config.data.dimension, hyp_len=config.data.hyp_len, hyp_gain=config.data.hyp_gain)

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                    else torch.device("cpu")
            )
        self.device = device
        self.num_timesteps = config.diffusion.num_diffusion_timesteps

        self.sde = VPSDE1D(schedule='cosine')

        self.dataset = dataset
        self.test_dataset = test_dataset

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        if args.distributed:
            sampler = DistributedSampler(self.dataset, shuffle=True,
                                     seed=args.seed if args.seed is not None else 0)
        else:
            sampler = None
        train_loader = data.DataLoader(
            self.dataset,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            sampler=sampler
        )

        # Model
        if config.model.model_type == "ddpm_mnist":
            model = Unet(dim=config.data.image_size,
                         channels=config.model.channels,
                         dim_mults=config.model.dim_mults,
                         is_conditional=config.model.is_conditional)
        elif config.model.model_type == "FNO":
            model = FNO(n_modes=config.model.n_modes, hidden_channels=config.model.hidden_channels, in_channels=config.model.in_channels, out_channels=config.model.out_channels,
                      lifting_channels=config.model.lifting_channels, projection_channels=config.model.projection_channels,
                      n_layers=config.model.n_layers, joint_factorization=config.model.joint_factorization,
                      norm=config.model.norm, preactivation=config.model.preactivation, separable=config.model.separable)
        elif config.model.model_type == "ddpm":
            model = Model(config)

        model = model.to(self.device)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[args.local_rank],)
                                                            #   find_unused_parameters=True)
        logging.info("Model loaded.")

        # Optimizer, LR scheduler
        optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)

        start_epoch, step = 0, 0

        for epoch in range(config.training.n_epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            data_start = time.time()
            data_time = 0

            for i, data in enumerate(train_loader):
                # MixedPolynomial 데이터셋의 경우 x, y, degree가 반환됨
                if config.data.dataset == 'MixedPolynomial':
                    x, y, label = data  # 차수는 원-핫 인코딩된 라벨로 제공됨
                    label = label.to(self.device)
                else:
                    x, y = data
                    label = None
                
                x = x.to(self.device).squeeze(-1)
                y = y.to(self.device).squeeze(-1)

                data_time += time.time() - data_start
                model.train()
                step += 1

                if config.data.dataset == 'Melbourne':
                    y = data_scaler(y)

                t = torch.rand(y.shape[0], device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                
                # 노이즈 생성
                e = self.W.sample(y.shape).to(self.device).squeeze(-1)
                
                # 조건부 모델인 경우 처리
                if hasattr(config.model, 'is_conditional') and config.model.is_conditional and label is not None:
                    # 원-핫 인코딩된 라벨을 입력에 결합
                    # 원본 함수 값이 채널 차원에서 첫 번째 채널이 되고, 
                    # 원-핫 인코딩된 라벨이 나머지 채널이 됨
                    y_with_cond = torch.cat([y.unsqueeze(1), label.unsqueeze(2).repeat(1, 1, y.shape[1])], dim=1)
                    
                    # 노이즈도 같은 차원 구조로 변환
                    e_with_cond = torch.cat([e.unsqueeze(1), torch.zeros(e.shape[0], label.shape[1], e.shape[1], device=self.device)], dim=1)
                    
                    # 조건부 손실 계산
                    loss = hilbert_loss_fn(model, self.sde, y_with_cond, t, e_with_cond).to(self.device)
                else:
                    # 기존 방식으로 손실 계산
                    loss = hilbert_loss_fn(model, self.sde, y, t, e).to(self.device)
                
                tb_logger.add_scalar("train_loss", torch.abs(loss), global_step=step)

                optimizer.zero_grad()
                loss.backward()

                if args.local_rank == 0:
                    logging.info(
                        f"step: {step}, loss: {torch.abs(loss).item()}, data time: {data_time / (i+1)}"
                    )

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass

                optimizer.step()

                if step % config.training.ckpt_store == 0:
                    self.ckpt_dir = os.path.join(args.log_path, 'ckpt.pth')
                    torch.save(model.state_dict(), self.ckpt_dir)

                data_start = time.time()

    def sample(self, score_model=None):
        args, config = self.args, self.config

        if config.model.model_type == "ddpm_mnist":
            model = Unet(dim=config.data.image_size,
                         channels=config.model.channels,
                         dim_mults=config.model.dim_mults,
                         is_conditional=config.model.is_conditional,)
        elif config.model.model_type == "FNO":
            model = FNO(n_modes=config.model.n_modes, hidden_channels=config.model.hidden_channels, in_channels=config.model.in_channels, out_channels=config.model.out_channels,
                      lifting_channels=config.model.lifting_channels, projection_channels=config.model.projection_channels,
                      n_layers=config.model.n_layers, joint_factorization=config.model.joint_factorization,
                      norm=config.model.norm, preactivation=config.model.preactivation, separable=config.model.separable)
        elif config.model.model_type == "ddpm":
            model = Model(config)

        model = model.to(self.device)

        if score_model is not None:
            model = score_model

        elif "ckpt_dir" in config.model.__dict__.keys():
            ckpt_dir = config.model.ckpt_dir
            states = torch.load(
                ckpt_dir,
                map_location=config.device,
            )

            if args.distributed:
                state_dict = OrderedDict()
                for k, v in states.items():
                    if 'module' in k:
                        name = k[7:]
                        state_dict[name] = v
                    else:
                        state_dict[k] = v

                model.load_state_dict(state_dict)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
            else:
                model.load_state_dict(states, strict=False)
        else:
            raise Exception("Fail to load model due to invalid ckpt_dir")

        logging.info("Done loading model")
        model.eval()

        test_loader = torch.utils.data.DataLoader(self.test_dataset, config.sampling.batch_size, shuffle=False)
        
        # 타겟 차수 결정
        # 명령줄 인수로 지정된 경우 우선 사용
        if args.target_degree is not None:
            target_degree = args.target_degree
        # 설정 파일에 지정된 경우 다음으로 사용
        elif hasattr(config.sampling, 'target_degree'):
            target_degree = config.sampling.target_degree
        # 기본값 사용 (4차 함수)
        else:
            target_degree = 4
        
        logging.info(f"샘플링 대상 차수: {target_degree}차 함수")

        if config.data.dataset == "MixedPolynomial":
            # MixedPolynomialDataset의 경우 x, y, degree가 반환됨
            x_0, y_0, labels = next(iter(test_loader))
            labels = labels.to(self.device)
            
            # 데이터셋에서 차수와 인덱스 매핑 정보 가져오기
            class_idx = self.test_dataset.get_index_from_degree(target_degree)
            
            # 타겟 차수에 해당하는 원-핫 인코딩 생성
            target_onehot = torch.zeros(config.sampling.batch_size, 4, device=self.device)
            target_onehot[:, class_idx] = 1.0
            
            # 매핑 - 실제 차수 변환 (색상 표시 등에 사용)
            degrees = torch.full((config.sampling.batch_size,), target_degree, device=self.device)
        else:
            # 다른 데이터셋의 경우 x, y만 반환됨
            x_0, y_0 = next(iter(test_loader))
            target_onehot = None

        if config.data.dataset in ['Quadratic', 'Quartic', 'Sextic', 'Octic', 'MixedPolynomial']:
            free_input = torch.rand((config.sampling.batch_size, y_0.shape[1])) * 20 - 10
            free_input = torch.sort(free_input)[0]

            # 데이터셋에 따른 차수 및 스케일 결정
            if config.data.dataset == 'Quadratic':
                degree = torch.full((config.sampling.batch_size,), 2, dtype=torch.long)
                scale_factor = 50
            elif config.data.dataset == 'Quartic':
                degree = torch.full((config.sampling.batch_size,), 4, dtype=torch.long)
                scale_factor = 50 * (4 / 2)
            elif config.data.dataset == 'Sextic':
                degree = torch.full((config.sampling.batch_size,), 6, dtype=torch.long)
                scale_factor = 50 * (6 / 2)
            elif config.data.dataset == 'Octic':
                degree = torch.full((config.sampling.batch_size,), 8, dtype=torch.long)
                scale_factor = 50 * (8 / 2)
            elif config.data.dataset == 'MixedPolynomial':
                # MixedPolynomial 데이터셋의 경우 타겟 차수 사용
                degree = degrees
                # 타겟 차수에 맞는 스케일 계산
                if target_degree == 2:
                    scale_factor = 50
                else:
                    scale_factor = 50 * (target_degree / 2)
                scale_factor = torch.full((config.sampling.batch_size,), scale_factor, device=self.device)

            # 타겟 차수에 맞는 다항식 함수 생성
            y00 = torch.zeros_like(free_input)
            
            # 타겟 차수의 계수 생성 (시드 고정)
            torch.manual_seed(42 + target_degree)
            
            # 계수 생성 및 계산
            coefficients = []
            for power in range(target_degree, -1, -2):
                if power % 2 == 0:
                    coef = torch.randint(low=0, high=2, size=(free_input.shape[0], 1)).repeat(1, free_input.shape[1]) * 2 - 1
                    coefficients.append((coef, power))
            
            # 다항식 계산 및 노이즈 추가
            for coef, power in coefficients:
                y00 += coef * (free_input ** power)
            
            # 노이즈 추가
            eps = torch.normal(mean=0., std=1., size=(free_input.shape[0], 1)).repeat(1, free_input.shape[1])
            y00 += eps

            with torch.no_grad():
                for _ in tqdm(range(1), desc="Generating image samples"):
                    y_shape = (config.sampling.batch_size, config.data.dimension)
                    t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T

                    # 초기 노이즈 샘플 생성
                    y_noise = self.W.free_sample(free_input).to(self.device) * self.sde.marginal_std(t)[:, None]
                    
                    # 조건부 모델인 경우 처리
                    if hasattr(config.model, 'is_conditional') and config.model.is_conditional and target_onehot is not None:
                        # 노이즈 샘플에 조건 결합
                        y_noise_with_cond = torch.cat([
                            y_noise.unsqueeze(1), 
                            target_onehot.unsqueeze(2).repeat(1, 1, y_noise.shape[1])
                        ], dim=1)
                        
                        # 커스텀 샘플링을 위한 함수 작성 필요
                        y = self._conditional_sampling(
                            y_noise_with_cond, 
                            model, 
                            free_input,
                            target_onehot
                        )
                    else:
                        # 기존 방식으로 샘플링
                        y = sampler(y_noise, model, self.sde, self.device, self.W, self.sde.eps, 
                                   config.data.dataset, sampler_input=free_input)

            # 스케일 적용
            y_0 = y_0 * scale_factor
            y = y * scale_factor

            _, ax = plt.subplots(1, 2, figsize=(10, 5))

            # MixedPolynomial 데이터셋은 차수별로 색상 구분하여 시각화
            if config.data.dataset == 'MixedPolynomial':
                colors = {2: 'blue', 4: 'green', 6: 'red', 8: 'purple'}
                
                # 모든 샘플이 동일한 차수를 가짐 (타겟 차수)
                color = colors[target_degree]
                
                # Ground truth 그래프 (타겟 차수와 일치하는 것만 그림)
                for i in range(config.sampling.batch_size):
                    ax[0].plot(x_0[i, :].cpu(), y_0[i, :].cpu(), color=color, alpha=0.7)
                
                # 생성된 샘플 그래프
                for i in range(y.shape[0]):
                    ax[1].plot(free_input[i, :].cpu(), y[i, :].cpu(), color=color, alpha=0.7)
                
                # 범례 추가
                import matplotlib.patches as mpatches
                legend_patches = [
                    mpatches.Patch(color=color, label=f'{target_degree}차 함수')
                ]
                ax[0].legend(handles=legend_patches)
                ax[1].legend(handles=legend_patches)
            else:
                # 기존 방식대로 시각화
                for i in range(config.sampling.batch_size):
                    ax[0].plot(x_0[i, :].cpu(), y_0[i, :].cpu())
                
                for i in range(y.shape[0]):
                    ax[1].plot(free_input[i, :].cpu(), y[i, :].cpu(), alpha=1)

            ax[0].set_title(f'Ground truth, len:{config.data.hyp_len:.2f}')

            n_tests = config.sampling.batch_size // 10
            
            print('Calculate Confidence Interval:')
            power_res = calculate_ci(y, y_0, n_tests=n_tests)
            print(f'Calculate Confidence Interval: resolution-free, power(avg of 30 trials): {power_res}')
            logging.info(f'Calculate Confidence Interval: resolution-free, power(avg of 30 trials): {power_res}')
            ax[1].set_title(f'resolution-free, power(avg of 30 trials): {power_res}')

        else:
            y_0 = y_0.squeeze(-1)
            with torch.no_grad():
                for _ in tqdm(range(1), desc="Generating image samples"):
                    y_shape = (config.sampling.batch_size, config.data.dimension)
                    t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T

                    y = self.W.sample(y_shape).to(self.device) * self.sde.marginal_std(t)[:, None]
                    y = sampler(y, model, self.sde, self.device, self.W,  self.sde.eps, config.data.dataset)

            _, ax = plt.subplots(1, 2, figsize=(10, 5))

            if config.data.dataset == 'Melbourne':
                lp = 10
                n_tests = y.shape[0] // 10
                y = data_inverse_scaler(y)
            if config.data.dataset == 'Gridwatch':
                lp = y.shape[0]
                n_tests = y.shape[0] // 10
                plt.ylim([-2, 3])

            for i in range(lp):
                ax[0].plot(x_0[i, :].cpu(), y[i, :].cpu())
                ax[1].plot(x_0[i, :].cpu(), y_0[i, :].cpu(), c='black', alpha=1)


            ax[0].set_title(f'Ground truth, len:{config.data.hyp_len:.2f}')

            for i in range(lp):
                ax[1].plot(x_0[i, :].cpu(), y[i, :].cpu(), alpha=1)


            power = calculate_ci(y, y_0, n_tests=n_tests)
            print(f'Calculate Confidence Interval: grid, 0th: {power}')

            ax[1].set_title(f'grid, power(avg of 30 trials):{power}')

        # Visualization figure save
        plt.savefig('visualization_default.png')
        print("Saved plot fig to {}".format('visualization_default.png'))
        plt.clf()
        plt.figure()

    # SDE 전이 함수 구현
    def transition(self, y, score, t, t_next):
        """SDE의 전이 함수 구현
        
        Args:
            y: 현재 샘플
            score: 모델이 예측한 점수
            t: 현재 시간
            t_next: 다음 시간
            
        Returns:
            mean: 다음 시간에서의 평균
            std: 다음 시간에서의 표준편차
        """
        sde = self.sde
        device = self.device
        
        # 기존 sampler에서 구현된 방식 참고
        beta_step = sde.beta(t) * (t - t_next)
        x_coeff = 1 + beta_step / 2.0
        
        score_term = score * torch.pow(sde.marginal_std(t), -(2.0 - 1))[:, None].to(device)
        score_coeff = beta_step
        
        mean = x_coeff[:, None].to(device) * y + score_coeff[:, None].to(device) * score_term
        
        noise_coeff = torch.pow(beta_step, 1 / 2.0)
        std = noise_coeff
        
        return mean, std
    
    # 조건부 샘플링을 위한 커스텀 함수
    def _conditional_sampling(self, y_noise_with_cond, model, free_input, condition):
        """조건부 모델을 위한 샘플링 함수
        
        Args:
            y_noise_with_cond: 조건이 결합된 초기 노이즈
            model: 확산 모델
            free_input: 입력 x 좌표
            condition: 원-핫 인코딩된 조건 (차수)
            
        Returns:
            생성된 샘플
        """
        config = self.config
        device = self.device
        sde = self.sde
        
        # 초기 시간 및 노이즈 분리
        t = torch.ones(y_noise_with_cond.shape[0], device=device) * sde.T
        y_t = y_noise_with_cond[:, 0]  # 첫 채널이 실제 함수 값
        
        # 역확산 과정 수행
        steps = config.diffusion.num_diffusion_timesteps
        timesteps = torch.linspace(sde.T, sde.eps, steps + 1).to(device)
        
        with torch.no_grad():
            for i in tqdm.tqdm(range(steps)):
                vec_s = torch.ones(y_t.shape[0], device=device) * timesteps[i]
                vec_t = torch.ones(y_t.shape[0], device=device) * timesteps[i + 1]
                
                # 조건을 함수 값과 결합
                y_with_cond = torch.cat([
                    y_t.unsqueeze(1),
                    condition.unsqueeze(2).repeat(1, 1, y_t.shape[1])
                ], dim=1)
                
                # 모델에서 노이즈 예측 값 얻기
                score = model(y_with_cond, vec_s)
                
                # 노이즈 추정치로 샘플 업데이트 (sde_score_update 함수와 유사)
                mean, std = self.transition(y_t, score, vec_s, vec_t)
                
                # W에서 노이즈 생성
                if free_input is not None:
                    noise = self.W.free_sample(free_input=free_input).to(device)
                else:
                    noise = self.W.sample(y_t.shape).to(device)
                
                # 다음 단계 샘플 생성
                y_t = mean + std[:, None].to(device) * noise
                
                # 샘플 클리핑 (sampler 함수와 동일)
                size = y_t.shape
                l = y_t.shape[0]
                y_t = y_t.reshape((l, -1))
                indices = y_t.norm(dim=1) > 10
                if config.data.dataset == 'Gridwatch':
                    y_t[indices] = y_t[indices] / y_t[indices].norm(dim=1)[:, None] * 17
                else:
                    y_t[indices] = y_t[indices] / y_t[indices].norm(dim=1)[:, None] * 10
                y_t = y_t.reshape(size)
                
        return y_t  # 최종 생성된 함수 값
