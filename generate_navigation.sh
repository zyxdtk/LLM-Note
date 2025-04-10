#!/bin/bash

# 项目根目录
root_dir="."

# 清空或创建 README.md 文件
> README.md

# 写入标题
echo "# 项目导航" >> README.md

# 递归遍历目录
find "$root_dir" -name "*.md" | while read -r file; do
    # 生成导航链接
    link="- [$(basename "$file")]($file)"
    # 将链接写入 README.md 文件
    echo "$link" >> README.md
done