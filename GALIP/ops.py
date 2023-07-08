import torch
import torch.nn.functional as F
import torch.nn as nn

from torchvision import transforms

def get_ratio(reg_every):
    if reg_every == 0:
        reg_ratio = 1
    else:
        reg_ratio = float(reg_every) / float(reg_every + 1)

    return reg_ratio

def apply_gradients(loss, optim, mixed_flag=False, scaler_x=None, scaler_min=None):
    optim.zero_grad()

    if mixed_flag:
        scaler_x.scale(loss).backward()
        scaler_x.step(optim)
        scaler_x.update()
        if scaler_x.get_scale() < scaler_min:
            scaler_x.update(16384.0)
    else:
        loss.backward()
        optim.step()

def moving_average(ema_model, origin_model, decay=0.999):
    # model1 = ema
    # model2 = origin

    with torch.no_grad():
        ema_param = dict(ema_model.named_parameters())
        origin_param = dict(origin_model.named_parameters())

        for k in ema_param.keys():
            ema_param[k].data.mul_(decay).add_(origin_param[k].data, alpha=1 - decay)
            # ema_param[k].data = decay * ema_param[k].data + (1 - decay) * origin_param[k].data

def d_hinge_loss(real_pred, fake_pred, fake_pred2):
    real_loss = torch.mean(F.relu(1.0 - real_pred))
    fake_loss = torch.mean(F.relu(1.0 + fake_pred))
    if fake_pred2 is None:
        d_loss = real_loss + fake_loss
    else:
        fake_loss2 = torch.mean(F.relu(1.0 + fake_pred2))
        fake_loss = (fake_loss + fake_loss2) * 0.5
        d_loss = real_loss + fake_loss
    return d_loss

def g_hinge_loss(fake_pred):
    g_loss = -torch.mean(fake_pred)
    return g_loss


def d_logistic_loss(real_pred, fake_pred, fake_pred2):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    if fake_pred2 is None:
        return real_loss.mean() + fake_loss.mean()
    else:
        fake_loss2 = F.softplus(fake_pred2)
        return real_loss.mean() + (fake_loss.mean() + fake_loss2.mean()) * 0.5

def d_r1_loss(logits, real_img, text_embed=None):
    if text_embed is None:
        grad_real = torch.autograd.grad(
            outputs=logits.sum(),
            inputs=real_img,
            create_graph=True,
        )[0]
        grad_penalty = (grad_real ** 2).reshape(grad_real.shape[0], -1).sum(1).mean()

    else:
        grads = torch.autograd.grad(
            outputs=logits.sum(),
            inputs=(real_img, text_embed),
            create_graph=True,
        )
        grad0 = grads[0].view(grads[0].size(0), -1)
        grad1 = grads[1].view(grads[1].size(0), -1)
        grad = torch.cat((grad0, grad1), dim=1)
        # norm은 torch.sqrt((grad ** 2).sum(1)) 임
        grad_penalty = (grad ** 2).sum(1).mean()

    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def d_adv_loss(real_pred, fake_pred, fake_pred2=None, gan_type='gan'):
    if gan_type == 'hinge':
        loss = d_hinge_loss(real_pred, fake_pred, fake_pred2)
    else:
        loss = d_logistic_loss(real_pred, fake_pred, fake_pred2)

    return loss

def g_adv_loss(fake_pred, gan_type='gan'):
    if gan_type == 'hinge':
        loss = g_hinge_loss(fake_pred)
    else:
        loss = g_nonsaturating_loss(fake_pred)

    return loss


def predict_loss(predictor, img_feature, text_feature, negtive):
    output = predictor(img_feature, text_feature)
    err = hinge_loss(output, negtive)
    return output,err

def hinge_loss(output, negtive):
    if negtive==False:
        err = torch.mean(F.relu(1. - output))
    else:
        err = torch.mean(F.relu(1. + output))
    return err

def MA_GP_FP32(img, sent, out):
    grads = torch.autograd.grad(outputs=out,
                            inputs=(img, sent),
                            grad_outputs=torch.ones(out.size()).cuda(),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0,grad1),dim=1)
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp = 2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp

def MA_GP_MP(img, sent, out, scaler):
    grads = torch.autograd.grad(outputs=scaler.scale(out),
                            inputs=(img, sent),
                            grad_outputs=torch.ones_like(out),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    inv_scale = 1./(scaler.get_scale()+float("1e-8"))
    #inv_scale = 1./scaler.get_scale()
    grads = [grad * inv_scale for grad in grads]
    with torch.cuda.amp.autocast():
        grad0 = grads[0].view(grads[0].size(0), -1)
        grad1 = grads[1].view(grads[1].size(0), -1)
        grad = torch.cat((grad0,grad1),dim=1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp

# clip loss
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


    x = denormalize(x)
    x = resize(x)
    x = zero_to_one(x)
    x = norm_mean_std(x)

    return x

def cosine_sim_loss(image_feat, text_feat):
    image_feat = image_feat / image_feat.norm(p=2, dim=-1, keepdim=True)
    text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)

    loss = -F.cosine_similarity(image_feat, text_feat).mean()
    return loss

def clip_score(clip_model, image, text):
    txt_features = clip_model.get_text_features(text)

    processed_image = clip_image_process(image)
    img_features = clip_model.get_image_features(processed_image)

    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
    txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

    # score = 100 * (img_features * txt_features).sum(axis=-1)
    # score = torch.mean(score)

    score = -F.cosine_similarity(img_features, txt_features).mean()

    return score

def clip_image_score(clip_model, image1, image2):
    processed_image1 = clip_image_process(image1)
    processed_image2 = clip_image_process(image2)

    img_features1 = clip_model.get_image_features(processed_image1)
    img_features2 = clip_model.get_image_features(processed_image2)

    img_features1 = img_features1 / img_features1.norm(p=2, dim=-1, keepdim=True)
    img_features2 = img_features2 / img_features2.norm(p=2, dim=-1, keepdim=True)

    # score = 100 * (img_features1 * img_features2).sum(axis=-1)
    # score = torch.mean(score)

    score = -F.cosine_similarity(img_features1, img_features2).mean()

    return score

def contrastive_loss(logits, dim) :
    neg_ce = torch.diag(nn.functional.log_softmax(logits, dim=dim))
    return -neg_ce.mean()

def clip_score_(clip_model, image, text):
    txt_features = clip_model.get_text_features(text)

    processed_image = clip_image_process(image)
    img_features = clip_model.get_image_features(processed_image)

    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
    txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale = clip_model.logit_scale.exp()
    similarity = torch.matmul(txt_features, img_features.t()) * logit_scale

    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)

    return (caption_loss + image_loss) / 2.0

def clip_image_score_(clip_model, image1, image2):
    processed_image1 = clip_image_process(image1)
    processed_image2 = clip_image_process(image2)

    img_features1 = clip_model.get_image_features(processed_image1)
    img_features2 = clip_model.get_image_features(processed_image2)

    img_features1 = img_features1 / img_features1.norm(p=2, dim=-1, keepdim=True)
    img_features2 = img_features2 / img_features2.norm(p=2, dim=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale = clip_model.logit_scale.exp()
    similarity = torch.matmul(img_features1, img_features2.t()) * logit_scale

    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)

    return (caption_loss + image_loss) / 2.0

def convert_to_billion_and_million(value, decimal_places=2):
    billion = round(value / 1_000_000_000, decimal_places)
    million = round(value / 1_000_000, decimal_places)

    return billion, million