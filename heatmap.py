import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn.functional as F

# โหลดโมเดล ResNet50 ที่ผ่านการฝึกบน ImageNet
model = resnet50(pretrained=True)
model.eval()

# คลาสสำหรับ Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook เพื่อดึงค่า Gradient และ Activation
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()

        # คำนวณ Gradient ของคลาสเป้าหมาย
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        # คำนวณ Weight และทำ Weighted Sum
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # Normalization
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.squeeze().cpu().detach().numpy()

# เลือก Layer ที่ต้องการใช้ Grad-CAM
grad_cam = GradCAM(model, model.layer4[-1])

# แปลงภาพเป็น Tensor
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# เปิดเว็บแคม
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงสีเป็น RGB และปรับขนาดให้เข้ากับโมเดล
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess_image(img_rgb)

    # สร้าง Attention Heatmap
    heatmap = grad_cam.generate_heatmap(input_tensor)

    # แปลง heatmap เป็นภาพ
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # รวมภาพต้นฉบับกับ heatmap
    output = cv2.hconcat([frame, heatmap])

    # แสดงผล
    cv2.imshow("Webcam with Attention Heatmap", output)

    # กด 'q' เพื่อออกจาก loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
