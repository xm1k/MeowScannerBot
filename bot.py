import torch
from torchvision import transforms
from PIL import Image
import telebot
from config import TOKEN

# --- device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- load models (map to device)
model = torch.load('trained_model.pth', weights_only=False, map_location=device)
model.eval()
model.to(device)

is_cat_model = torch.load('./is_cat/is_cat_model.pth', weights_only=False, map_location=device)
is_cat_model.eval()
is_cat_model.to(device)

# optional: load ViT if exists
try:
    vit_model = torch.load('vit_full.pth', weights_only=False, map_location=device)
    vit_model.eval()
    vit_model.to(device)
    has_vit = True
except Exception as e:
    print("ViT model not loaded:", e)
    vit_model = None
    has_vit = False

cat_names = [
    "Абиссинская","Ангорская","Балинезийская","Бенгальская","Бомбейская",
    "Британская короткошёрстная","Бирманская","Шартрез","Европейская короткошёрстная",
    "Японский бобтейл","Корат","Мейн-кун","Невская маскарадная","Норвежская лесная",
    "Персидская","Рэгдолл","Рекс","Русская голубая","Саванна","Шотландская вислоухая",
    "Сиамская","Сингапурская","Сфинкс"
]

target_size = (256, 256)

# preprocess for main model and is_cat (256x256)
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    sides_ratio = target_size[0] / target_size[1]

    if width / height > sides_ratio:
        new_width = int(height * sides_ratio)
        left = (width - new_width) / 2
        right = left + new_width
        image = image.crop((left, 0, right, height))
    else:
        new_height = int(width / sides_ratio)
        top = (height - new_height) / 2
        bottom = top + new_height
        image = image.crop((0, top, width, bottom))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # shape (1,C,H,W)

# preprocess for ViT (224x224)
def preprocess_image_vit(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def main(message):
    bot.send_message(message.chat.id, "Привет! Отправь фото кошки — скажу породу.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    image_path = "received_photo.jpg"
    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    bot.send_message(message.chat.id, "Анализ 🔎")
    try:
        # prepare inputs
        input_image = preprocess_image(image_path).to(device)        # for ResNet / is_cat
        if has_vit:
            vit_input = preprocess_image_vit(image_path).to(device)  # for ViT

        with torch.no_grad():
            # is_cat check
            is_cat_out = torch.softmax(is_cat_model(input_image), dim=1)
            # assuming class 1 is "cat"
            cat_probability = float(is_cat_out[0, 1].cpu().item())

            if cat_probability < 0.5:
                response = "На фото не обнаружена кошка или кошку плохо видно😿. Пожалуйста, отправьте другое изображение."
            else:
                # --- ResNet predictions
                out_res = model(input_image)
                probs_res = torch.softmax(out_res, dim=1)[0]
                top_probs_r, top_classes_r = torch.topk(probs_res, 3)

                resnet_preds = []
                for i in range(top_probs_r.size(0)):
                    cls_idx = int(top_classes_r[i].item())
                    prob = float(top_probs_r[i].item()*100)
                    resnet_preds.append(f"{cat_names[cls_idx]}: {prob:.2f}%")

                # --- ViT predictions (if available)
                vit_preds = None
                if has_vit:
                    out_vit = vit_model(vit_input)
                    probs_vit = torch.softmax(out_vit, dim=1)[0]
                    top_probs_v, top_classes_v = torch.topk(probs_vit, 3)

                    vit_preds = []
                    for i in range(top_probs_v.size(0)):
                        cls_idx = int(top_classes_v[i].item())
                        prob = float(top_probs_v[i].item()*100)
                        vit_preds.append(f"{cat_names[cls_idx]}: {prob:.2f}%")

                # build response
                response_lines = []
                response_lines.append("🐱 ResNet50:")
                response_lines.extend(resnet_preds)
                if vit_preds is not None:
                    response_lines.append("\n🤖 ViT:")
                    response_lines.extend(vit_preds)

                response = "\n".join(response_lines)

        bot.send_message(message.chat.id, response)
    except Exception as exc:
        bot.send_message(message.chat.id, "Ошибка при обработке изображения.")
        # для дебага можно логировать подробности:
        print("Handle photo error:", exc)

bot.polling(none_stop=True)
