import os
import io
from io import BytesIO
import base64

try:
    import requests
    requests_available = True
except ImportError:
    print("警告: 无法导入requests库。请使用 'pip install requests' 安装。")
    requests_available = False

try:
    from PIL import Image
    # 尝试导入HEIF插件
    heif_available = False
    try:
        from PIL import HeifImagePlugin
        heif_available = True
        print("成功加载PIL的HEIF插件，可以处理HEIF格式图片。")
        # 显式使用插件以避免未使用警告
        if hasattr(HeifImagePlugin, 'register'):
            HeifImagePlugin.register()
    except ImportError:
        # 尝试直接导入pillow_heif
        try:
            import pillow_heif
            heif_available = True
            print("成功导入pillow_heif库，可以处理HEIF格式图片。")
            # 注册HEIF格式
            pillow_heif.register_heif_opener()
        except ImportError:
            print("提示: 未找到HEIF插件，可能无法处理HEIF格式图片。")
            print("       已安装pillow-heif但仍有问题，请尝试更新PIL和pillow-heif。")
    pil_available = True
except ImportError:
    print("警告: 无法导入PIL库。请使用 'pip install Pillow' 安装。")
    pil_available = False


def convert_image_to_jpeg(image_path):
    """将图片转换为JPEG格式（支持HEIF格式）
    Args:
        image_path: 图片文件路径

    Returns:
        bytes: 转换后的JPEG图片数据或原始图片数据（如果已为JPEG）
    """
    try:
        # 检查文件扩展名是否为HEIF/HEIC
        file_ext = os.path.splitext(image_path)[1].lower()
        is_heif = file_ext in ['.heif', '.heic']

        if is_heif:
            print(f"检测到HEIF/HEIC格式图片: {image_path}")
            # 特殊处理HEIF格式
            if not pil_available:
                print("错误: PIL库不可用，无法处理HEIF图片。")
                return None
            if not heif_available:
                print("错误: HEIF插件不可用，无法处理HEIF图片。")
                print("       请确保已安装pillow-heif: pip install pillow-heif")
                return None

        # 打开图片文件
        with Image.open(image_path) as img:
            print(f"图片格式: {img.format}, 模式: {img.mode}")
            # 检查是否需要转换
            needs_conversion = img.format.lower() != 'jpeg' or img.mode == 'rgba'

            if not needs_conversion:
                # 不需要转换，直接返回原始图片数据
                with open(image_path, 'rb') as f:
                    return f.read()
            else:
                # 如果是RGBA模式，转换为RGB
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                # 创建字节流缓冲区
                buffer = io.BytesIO()
                # 保存为JPEG格式到缓冲区
                img.save(buffer, format="JPEG", quality=95)
                # 返回缓冲区内容
                return buffer.getvalue()
    except Exception as e:
        print(f"转换图片格式时出错: {e}")
        # 对于HEIF格式，提供更具体的错误信息
        if is_heif:
            print("提示: 处理HEIF格式图片时出错。")
            print("       请确保已正确安装pillow-heif并更新到最新版本。")
            print("       可以尝试: pip install --upgrade pillow-heif")
        return None


def encode_image(image_data, max_width=800, max_height=600, quality=85):
    """将图片数据编码为base64格式，并在编码前压缩图像分辨率

    Args:
        image_data: 图片字节数据
        max_width: 最大宽度，超过此宽度将被缩放
        max_height: 最大高度，超过此高度将被缩放
        quality: JPEG保存质量 (1-100)

    Returns:
        str: base64编码的图片数据
    """
    try:
        # 检查PIL是否可用
        if not pil_available:
            print("错误: PIL库不可用，无法压缩图像。")
            # 直接编码原始数据
            return base64.b64encode(image_data).decode("utf-8")
        
        # 将字节数据转换为PIL Image对象
        image = Image.open(BytesIO(image_data))
        
        # 获取原始尺寸
        width, height = image.size
        
        # 计算新尺寸，保持宽高比
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # 调整图像大小
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # 将调整后的图像保存到缓冲区
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        
        # 编码为base64
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"编码图片时出错: {e}")
        return None


# Ollama VLM 图像描述示例
# 此脚本演示如何使用Ollama API发送带本地图片的请求
def generate_image_description(url, image_path, model="qwen2.5vl:7b", prompt="描述这张图"):
    """使用Ollama API生成图片描述
    
    参数:
    url -- API端点URL
    image_path -- 图片文件路径
    model -- 使用的模型名称
    prompt -- 提示文本
    
    返回:
    str -- 生成的图片描述，如果发生错误则返回None
    """
    # 检查必要的库是否可用
    if not requests_available:
        print("错误: requests库不可用，无法发送API请求。")
        return None
    
    if not pil_available:
        print("错误: PIL库不可用，无法处理图片。")
        return None

    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在: {image_path}")
        return None

    # 转换图片格式为JPEG
    image_data = convert_image_to_jpeg(image_path)
    if not image_data:
        print("图片格式转换失败，无法继续")
        return None

    # 编码图片为base64格式
    base64_image = encode_image(image_data)
    if not base64_image:
        print("图片编码失败，无法继续")
        return None

    print(base64_image)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # 非流式响应
        "images": [base64_image]  # 使用base64编码的图片
    }

    # 发送POST请求
    response = None
    try:
        print("正在向Ollama API发送请求...")
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()  # 检查请求是否成功

        # 解析JSON响应
        result = response.json()
        description = result.get("response", "未找到响应内容")
        print("模型生成结果：", description)
        return description

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        if response is not None:
            print(f"响应状态码: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"错误详情: {error_detail}")
            except Exception:
                print("无法解析错误响应")
    except Exception as e:
        print(f"处理响应时发生错误: {e}")

    return None


def main():
    # 请求参数
    url = "http://localhost:11434/api/generate"
    image_path = '/Users/ly/Documents/private/pic/baby_photo/0c81956738081221b06bf21b2411e246.JPG'
    model = "qwen2.5vl:7b"
    prompt = "描述这张图"

    # 生成图片描述
    # 生成图片描述
    description = generate_image_description(url, image_path, model, prompt)
    print(f"图片描述: {description}")

    print("程序执行完毕")


if __name__ == "__main__":
    main()