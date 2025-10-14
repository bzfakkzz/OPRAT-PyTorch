import time

from attacks import create_pytorch_classifier, generate_adversarial_samples
from data import generate_random_data
from models import TorchVGG16

start = time.perf_counter()
attack='Universal'
test_data = generate_random_data((3,224,224), 10)  # 测试数据
torch_model = TorchVGG16(num_classes=100)

print(f"现在攻击的是{attack}....")
#D_attack ←- DataAttack(D_test, aj)
#x = test_data.numpy()
classifier=create_pytorch_classifier(torch_model,(3,224,224),100)
attack_data=generate_adversarial_samples(attack,classifier,test_data)

end = time.perf_counter()
print(f"运行时间: {end - start:.6f} 秒")