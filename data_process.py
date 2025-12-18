import pandas as pd
import re
from collections import defaultdict
import os

def diagnose_file_format(file_path):
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            first_500_chars = file.read(500)
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                first_500_chars = file.read(500)
        except Exception as e:
            
            return
    
    
    
    # 检查常见的分隔符模式
    lines = first_500_chars.split('\n')[:20]  
    
    for i, line in enumerate(lines):
        print(f"行 {i+1}: {repr(line)}")

def parse_wos_files_advanced(file_paths):
   
    records = []
    
    for file_path in file_paths:
        print(f"\n正在处理文件: {file_path}")
        
        # 先诊断文件格式
        diagnose_file_format(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
            except Exception as e:
                
                continue
        
        # 方法1: 尝试按记录分隔符分割
        record_blocks = []
        
        # 尝试不同的记录分隔符
        separators = [
            r'\n\s*ER\s*\n',  # 标准WoS格式
            r'\n\s*EF\s*\n',  # 可能的结束标记
            r'\n\n+',         # 多个空行
            r'\n\.\n',        # 点号分隔
        ]
        
        for separator in separators:
            record_blocks = re.split(separator, content)
            if len(record_blocks) > 1:
                print(f"使用分隔符 '{separator}' 找到 {len(record_blocks)} 个记录块")
                break
        else:
          
            record_blocks = [content]
        
        for block_index, block in enumerate(record_blocks):
            if not block.strip():
                continue
                
            record_data = {}
            lines = block.strip().split('\n')
            
            current_field = None
            current_value = []
            
            for line_index, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # 多种字段识别模式
                field_patterns = [
                    (r'^([A-Z][A-Z])\s+(.*)$', 2),  # 标准模式: "XX 内容"
                    (r'^([A-Z][A-Z])-(.*)$', 2),    # 变体模式: "XX-内容"
                    (r'^([A-Z])\s+(.*)$', 1),       # 单字母字段: "X 内容"
                ]
                
                matched = False
                for pattern, field_length in field_patterns:
                    match = re.match(pattern, line)
                    if match:
                        # 保存上一个字段
                        if current_field and current_value:
                            record_data[current_field] = ' '.join(current_value).strip()
                        
                        # 开始新字段
                        current_field = match.group(1)
                        current_value = [match.group(2).strip()]
                        matched = True
                        break
                
                if not matched and current_field:
                    # 续行内容
                    current_value.append(line)
            
            # 保存最后一个字段
            if current_field and current_value:
                record_data[current_field] = ' '.join(current_value).strip()
            
            # 验证记录是否有效
            if is_valid_record(record_data):
                records.append(record_data)
            else:
                print(f"跳过无效记录块 {block_index} (缺少必要字段)")
    
    
    return records

def is_valid_record(record_data):
    
    #检查记录是否包含必要字段
    
    # 至少要有作者或标题字段
    has_essential_field = 'AU' in record_data or 'TI' in record_data
    
    if has_essential_field:
        # 调试输出一些样本记录
        if len(record_data) > 0:
            print(f"有效记录样本 - 字段: {list(record_data.keys())}")
    
    return has_essential_field

def debug_record_parsing(records):
    """
    调试记录解析结果
    """
    
    print(f"总记录数: {len(records)}")
    
    if records:
        
        for i, record in enumerate(records[:3]):
            print(f"记录 {i+1}:")
            for key, value in list(record.items())[:5]:  
                print(f"  {key}: {value[:100]}{'...' if len(value) > 100 else ''}")
            print()
        
        # 统计字段出现频率
        field_counts = defaultdict(int)
        for record in records:
            for field in record.keys():
                field_counts[field] += 1
        
        
        for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {field}: {count} 次")

def extract_authors_and_year_with_debug(records):
    """
    带调试信息的作者提取函数
    """
    edges_data = []
    author_mapping = {}
    author_counter = 0
    paper_counter = 0
    
    
    
    for record_index, record in enumerate(records):
        # 提取作者
        authors = []
        if 'AU' in record:
            author_text = record['AU']
            print(f"记录 {record_index} 作者原始文本: {repr(author_text[:100])}")
            
            # 多种作者分隔符尝试
            separators = [';', '\n', '|', ',', ' and ']
            author_list = []
            
            for sep in separators:
                author_list = [a.strip() for a in author_text.split(sep) if a.strip()]
                if len(author_list) > 1:
                    print(f"  使用分隔符 '{sep}' 找到 {len(author_list)} 个作者")
                    break
            else:
                # 如果没有找到分隔符，尝试其他方法
                author_list = [author_text]
            
            authors = author_list
        
        # 调试输出
        if authors:
            print(f"  解析出的作者: {authors}")
        else:
            
            continue
        
        # 提取年份
        year = 'Unknown'
        year_fields = ['PY', 'PD', 'DA', 'YR']
        for field in year_fields:
            if field in record:
                year_match = re.search(r'(\d{4})', record[field])
                if year_match:
                    year = year_match.group(1)
                    break
        
        print(f"  年份: {year}")
        
        # 为每个作者分配唯一ID
        author_ids = []
        for author in authors:
            normalized_author = normalize_author_name(author)
            
            if normalized_author not in author_mapping:
                author_mapping[normalized_author] = author_counter
                author_counter += 1
                print(f"  新作者: {author} -> {normalized_author} (ID: {author_mapping[normalized_author]})")
            
            author_ids.append(author_mapping[normalized_author])
        
        # 生成合作边
        if len(author_ids) >= 2:
            print(f"  生成 {len(author_ids)} 个作者之间的合作边")
            for i in range(len(author_ids)):
                for j in range(i + 1, len(author_ids)):
                    edges_data.append({
                        'node_id_1': author_ids[i],
                        'node_id_2': author_ids[j],
                        'weight': 1,
                        'year': year,
                        'paper_id': paper_counter
                    })
            
            paper_counter += 1
        else:
            print(f"  只有 {len(author_ids)} 个作者，跳过合作边生成")
    
    print(f"\n作者提取完成:")
    print(f"  唯一作者数: {len(author_mapping)}")
    print(f"  合作边数: {len(edges_data)}")
    
    reverse_mapping = {v: k for k, v in author_mapping.items()}
    return edges_data, reverse_mapping

def normalize_author_name(author_name):
    #标准化作者名
    
    # 移除多余空格
    author_name = re.sub(r'\s+', ' ', author_name.strip())
    
    # 处理常见的姓名格式变体
    author_name = re.sub(r',\s*', ' ', author_name)  # 移除逗号
    author_name = re.sub(r'\.', '', author_name)     # 移除点号
    
    return author_name.upper()

def create_output_files(edges_data, author_mapping, output_dir='output'):
    
    #创建输出文件
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建节点文件
    nodes_data = []
    for node_id, author_name in author_mapping.items():
        nodes_data.append({
            'node_id': node_id,
            'node_name': author_name
        })
    
    nodes_df = pd.DataFrame(nodes_data)
    nodes_path = os.path.join(output_dir, 'nodes.csv')
    nodes_df.to_csv(nodes_path, index=False, encoding='utf-8')
    print(f"共 {len(nodes_data)} 个作者节点")
    
    # 创建边文件
    if edges_data:
        edges_df = pd.DataFrame(edges_data)
        
        # 合并相同作者对的权重
        edges_aggregated = edges_df.groupby(['node_id_1', 'node_id_2', 'year']).agg({
            'weight': 'sum'
        }).reset_index()
        
        edges_path = os.path.join(output_dir, 'edges.csv')
        edges_aggregated.to_csv(edges_path, index=False, encoding='utf-8')
        print(f"边文件已保存: {edges_path}")
        print(f"  共 {len(edges_aggregated)} 条合作边")
        
        return nodes_df, edges_aggregated
    else:
        
        # 仍然创建空的边文件以便后续处理
        empty_edges = pd.DataFrame(columns=['node_id_1', 'node_id_2', 'weight', 'year'])
        edges_path = os.path.join(output_dir, 'edges.csv')
        empty_edges.to_csv(edges_path, index=False, encoding='utf-8')
        
        
        return nodes_df, empty_edges

def main():
    # 文件路径
    data_dir = 'data'
    file_paths = [
        os.path.join(data_dir, 'savedrecs 1-1000.txt'),
        os.path.join(data_dir, 'savedrecs 1001-2000.txt'),
        os.path.join(data_dir, 'savedrecs 2001-3000.txt'),
        os.path.join(data_dir, 'savedrecs 3001-3529.txt')
    ]
    
    
    # 解析文件
    records = parse_wos_files_advanced(file_paths)
    
    # 调试记录解析
    debug_record_parsing(records)
    
    # 提取作者和合作信息
    edges_data, author_mapping = extract_authors_and_year_with_debug(records)
    
    # 创建输出文件
    nodes_df, edges_df = create_output_files(edges_data, author_mapping)
    
    
if __name__ == "__main__":
    main()