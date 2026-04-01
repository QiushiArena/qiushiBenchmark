import json

def main():
    # 该脚本将data.json中的数据转化成指定格式
    with open("dataset/benchmark_data/iterate.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    transformed_data = []
    
    # 分别处理4组数据，每组4*625
    id = 1
    for i in range(4):
        # 每组数据使用对应的 llm_text_{i+1}
        group_data = data[i*625:(i+1)*625] + data[2500+i*625:2500+(i+1)*625] + data[5000+i*625:5000+(i+1)*625] + data[7500+i*625:7500+(i+1)*625]
        llm_key = f"llm_text_{i+1}"
        
        for item in group_data:
            transformed_item = {
                "id": id,
                "domain": item["domain"],
                "iterated_text": item[llm_key],
                "llm_times": (i+1) * 4
            }
            transformed_data.append(transformed_item)
            id += 1
    
    # 输出结果
    output_file = "dataset/benchmark_data/iter.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)
    
    print(f"sucess, output file: {output_file}")

if __name__ == "__main__":
    main()