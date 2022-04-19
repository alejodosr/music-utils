import torch


class AnimeFaceChanger:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator",  device=device).eval()
        self.face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device)

    def change_face(self, face_img):
        return self.face2paint(self.model, face_img, side_by_side=False)
