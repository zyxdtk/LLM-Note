import os

def generate_tree(directory, prefix='', is_last=True, exclude_dirs=['.git', '.github']):
    """递归生成目录树结构"""
    indent = '    ' if is_last else '│   '
    new_prefix = prefix + ('└── ' if is_last else '├── ')
    
    items = []
    try:
        items = sorted(os.listdir(directory))
    except PermissionError:
        return ""
    
    dirs = [d for d in items if os.path.isdir(os.path.join(directory, d)) and d not in exclude_dirs]
    files = [f for f in items if os.path.isfile(os.path.join(directory, f)) and f.endswith('.md')]
    
    tree = []
    for i, item in enumerate(dirs + files):
        is_last_item = i == len(dirs + files) - 1
        path = os.path.join(directory, item)
        
        if item in dirs:
            tree.append(f"{new_prefix}{item}/")
            subtree = generate_tree(path, prefix + indent, is_last_item, exclude_dirs)
            tree.append(subtree)
        else:
            tree.append(f"{new_prefix}[{item}]({path})")
    
    return '\n'.join(filter(None, tree))

# 生成目录树
tree_structure = generate_tree('.')
tree_structure = "# 项目目录树\n\n```\n" + tree_structure + "\n```"

# 写入README.md
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(tree_structure)