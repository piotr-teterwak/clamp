import open_clip
import torch.nn.functional as F

def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu"):
    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer


model, transform, tokenizer = load_open_clip("clamp", pretrained='/projectnb/ivc-ml/piotrt/checkpoints/final_checkpoint.pt')

classname= "insect"
device="cpu"

texts= "USER: A photo of a {}.Being brief, an image of {} has the following visual attributes:\n ASSISTANT:1. ".format(classname,classname)
texts = tokenizer(texts).to(device)  # tokenize
text_features = model.encode_text(texts)


images = torch.zeros(1,3,224,224)
image_features = model.encode_image(images)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

similarity = (image_features @ text_features.T)
