import torch
import torch.nn.functional as F

class PolynomialDataset(torch.utils.data.Dataset):
    def __init__(self, num_data, num_points, degree=2, seed=42):
        super().__init__()

        self.num_data = num_data
        self.num_points = num_points
        self.degree = degree
        self.seed = seed
        self.x = torch.linspace(start=-10., end=10., steps=self.num_points).unsqueeze(0).repeat(self.num_data, 1)
        self.dataset = self._create_dataset()

    def _create_dataset(self):
        torch.manual_seed(self.seed)
        
        # 모든 계수에 대해 랜덤 값 생성
        coefficients = []
        for power in range(self.degree, -1, -2):  # degree, degree-2, ..., 2, 0
            if power % 2 == 0:  # 짝수 차수만 사용 (x=0 기준 대칭)
                coef = torch.randint(low=0, high=2, size=(self.x.shape[0], 1)).repeat(1, self.num_points) * 2 - 1
                coefficients.append((coef, power))
        
        # 노이즈 생성
        eps = torch.normal(mean=0., std=1., size=(self.x.shape[0], 1)).repeat(1, self.num_points)
        
        # 다항식 계산
        y = torch.zeros_like(self.x)
        for coef, power in coefficients:
            y += coef * (self.x ** power)
        
        # 노이즈 추가
        y += eps
        
        # 높은 차수의 함수는 값이 매우 커질 수 있으므로 정규화 적용
        if self.degree > 2:
            scale_factor = 50 * (self.degree / 2)
        else:
            scale_factor = 50
            
        return y / scale_factor

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        return self.x[idx, :].unsqueeze(-1), self.dataset[idx, :].unsqueeze(-1)


class QuarticDataset(PolynomialDataset):
    """x=0을 기준으로 대칭인 4차 함수 데이터셋"""
    def __init__(self, num_data, num_points, seed=42):
        super().__init__(num_data, num_points, degree=4, seed=seed)


class SexticDataset(PolynomialDataset):
    """x=0을 기준으로 대칭인 6차 함수 데이터셋"""
    def __init__(self, num_data, num_points, seed=42):
        super().__init__(num_data, num_points, degree=6, seed=seed)


class OcticDataset(PolynomialDataset):
    """x=0을 기준으로 대칭인 8차 함수 데이터셋"""
    def __init__(self, num_data, num_points, seed=42):
        super().__init__(num_data, num_points, degree=8, seed=seed)


class MixedPolynomialDataset(torch.utils.data.Dataset):
    """2차, 4차, 6차, 8차 다항식이 혼합된 데이터셋"""
    def __init__(self, num_data, num_points, seed=42):
        super().__init__()

        self.num_data = num_data
        self.num_points = num_points
        self.seed = seed
        self.x = torch.linspace(start=-10., end=10., steps=self.num_points).unsqueeze(0).repeat(self.num_data, 1)
        
        # 각 차수별 샘플 수 계산 (균등 분포)
        self.samples_per_degree = num_data // 4
        
        # 각 차수별 데이터셋 생성
        torch.manual_seed(seed)
        self.quadratic_data = self._create_polynomial_data(degree=2, count=self.samples_per_degree)
        self.quartic_data = self._create_polynomial_data(degree=4, count=self.samples_per_degree)
        self.sextic_data = self._create_polynomial_data(degree=6, count=self.samples_per_degree)
        self.octic_data = self._create_polynomial_data(degree=8, count=self.samples_per_degree)
        
        # 모든 데이터 결합
        self.dataset = torch.cat([
            self.quadratic_data,
            self.quartic_data,
            self.sextic_data,
            self.octic_data
        ], dim=0)
        
        # 데이터와 차수 관계 추적 (차수를 클래스 인덱스로 매핑)
        # 차수 -> 클래스 인덱스 매핑: 2->0, 4->1, 6->2, 8->3
        self.degree_map = []
        self.degree_map.extend([0] * self.samples_per_degree)  # 2차 함수의 인덱스: 0
        self.degree_map.extend([1] * self.samples_per_degree)  # 4차 함수의 인덱스: 1
        self.degree_map.extend([2] * self.samples_per_degree)  # 6차 함수의 인덱스: 2
        self.degree_map.extend([3] * self.samples_per_degree)  # 8차 함수의 인덱스: 3
        self.degree_map = torch.tensor(self.degree_map)
        
        # 인덱스→차수 변환을 위한 매핑 배열
        self.index_to_degree = {0: 2, 1: 4, 2: 6, 3: 8}
        self.degree_to_index = {2: 0, 4: 1, 6: 2, 8: 3}
        
        # 클래스 수
        self.num_classes = 4

    def _create_polynomial_data(self, degree, count):
        """지정된 차수의 다항식 데이터를 생성"""
        # 임시 시드 생성 (차수별로 다른 시드 사용)
        temp_seed = self.seed + degree 
        torch.manual_seed(temp_seed)
        
        # x 데이터
        x = self.x[:count, :]
        
        # 계수 생성
        coefficients = []
        for power in range(degree, -1, -2):
            if power % 2 == 0:  # 짝수 차수만 사용 (x=0 기준 대칭)
                coef = torch.randint(low=0, high=2, size=(count, 1)).repeat(1, self.num_points) * 2 - 1
                coefficients.append((coef, power))
        
        # 노이즈
        eps = torch.normal(mean=0., std=1., size=(count, 1)).repeat(1, self.num_points)
        
        # 다항식 계산
        y = torch.zeros_like(x)
        for coef, power in coefficients:
            y += coef * (x ** power)
        
        # 노이즈 추가
        y += eps
        
        # 스케일링
        if degree > 2:
            scale_factor = 50 * (degree / 2)
        else:
            scale_factor = 50
            
        return y / scale_factor

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # 데이터 및 클래스 인덱스 가져오기
        x = self.x[idx, :].unsqueeze(-1)
        y = self.dataset[idx, :].unsqueeze(-1)
        class_idx = self.degree_map[idx]
        
        # 원-핫 인코딩 생성
        label_onehot = F.one_hot(class_idx, num_classes=self.num_classes).float()
        
        return x, y, label_onehot
        
    def get_degree_from_index(self, index):
        """클래스 인덱스로부터 실제 차수를 반환"""
        return self.index_to_degree[index.item()]
        
    def get_index_from_degree(self, degree):
        """차수로부터 클래스 인덱스를 반환"""
        return self.degree_to_index[degree] 