import os

# 项目根目录
root_dir = '.'

# 存储导航内容
navigation = []

# 递归遍历目录
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.md'):
            # 获取文件的相对路径
            file_path = os.path.join(root, file)
            # 生成导航链接
            link = f"- [{file}]({file_path})"
            navigation.append(link)

# 将导航内容写入 README.md 文件
with open('README.md', 'w', encoding='utf-8') as readme_file:
    readme_file.write("# 项目导航\n")
    for nav in navigation:
        readme_file.write(nav + '\n')