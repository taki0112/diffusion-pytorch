from torchmetrics.multimodal import CLIPScore
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPModel
import torch
from torchvision import transforms
import torch.nn.functional as F


def clip_image_process(x):
    def denormalize(x):
        # [-1, 1] ~ [0, 255]
        x = ((x + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

        return x

    def resize(x):
        x = transforms.Resize(size=[224, 224], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(x)
        return x

    def zero_to_one(x):
        x = x.float() / 255.0
        return x

    def norm_mean_std(x):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        x = transforms.Normalize(mean=mean, std=std, inplace=True)(x)
        return x

    # 만약 x가 [-1, 1] 이면, denorm을 해줍니다.
    # x = denormalize(x)
    x = resize(x)
    x = zero_to_one(x)
    x = norm_mean_std(x)

    return x

def clip_score(img_features, txt_features):
    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
    txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

    # score = 100 * (img_features * txt_features).sum(axis=-1)
    # score = torch.mean(score)

    # 위와 같다.
    score = F.cosine_similarity(img_features, txt_features)
    return score

# library
image = torch.randint(255, (2, 3, 224, 224))
text = ["a photo of a cat", "a photo of a cat"]
version = 'openai/clip-vit-large-patch14'
metric = CLIPScore(model_name_or_path=version)
score = metric(image, text)
print(score)

"""
Step 1. Model Init
"""
tokenizer = CLIPTokenizer.from_pretrained(version)
clip_text_encoder = CLIPTextModel.from_pretrained(version)
clip_image_encoder = CLIPVisionModel.from_pretrained(version)
clip_model = CLIPModel.from_pretrained(version)

"""
Step 2. Text
"""
batch_encoding = tokenizer(text, truncation=True, max_length=77, padding="max_length", return_tensors="pt")
# [input_ids, attention_mask] -> 둘다 [bs,77]의 shape을 갖고있습니다.
# input_ids는 주어진 텍스트를 토크나이즈한것이고, mask는 어디까지만이 유효한 token인지 알려줍니다. 1=유효, 0=의미없음

text_token = batch_encoding["input_ids"]
t = clip_text_encoder(text_token) # 이것은 clip_model.text_model(text_token)과 같다.
# [last_hidden_state, pooler_output] -> [bs, 77, 768], [bs, 768]
# last_hidden_state = word embedding
# pooler_output = sentence embedding

text_feature = clip_model.get_text_features(text_token)
# pooler_output(sentence embedding) 에 Linear를 태운것
# [bs, 768]

"""
Step 3. Image
"""
image = clip_image_process(image)
feat = clip_image_encoder(image) # 이것은 clip_model.vision_model(image)과 같다.
# [last_hidden_state, pooler_output] -> [bs, 256, 1024], [bs, 1024]

image_feature = clip_model.get_image_features(image)
# pooler_output에 Linear을 태운것

print(clip_score(image_feature, text_feature))

