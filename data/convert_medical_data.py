import json
import os

def preprocess_medical_data(input_file, output_file):
    # 读取原始JSON数据
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 创建新的数据集
    processed_data = []
    
    # 处理每个患者记录
    for patient in raw_data:
        # 提取患者信息
        gender = patient.get("患者基本信息", {}).get("性别", "")
        age = patient.get("患者基本信息", {}).get("年龄", "")
        chief_complaint = patient.get("主诉", "")
        present_illness = patient.get("现病史", "")
        past_medical_history = patient.get("既往史", "")
        personal_history = patient.get("个人史", "")
        # family_history = patient.get("家族史", "")  # 注意：原始数据可能没有此字段
        
        # 处理体格检查信息
        physical_examination = ""
        if "查体" in patient:
            exam_data = patient["查体"]
            if "生命体征" in exam_data:
                physical_examination += exam_data["生命体征"] + " "
            if "体格检查" in exam_data:
                physical_examination += exam_data["体格检查"]
        
        # 获取诊断结果作为输出
        diagnoses = patient.get("初步诊断", [])
        output = ", ".join(diagnoses) if diagnoses else ""
        output = "["+output+"]"
        
        # 创建指令文本
        instruction = f"""# 角色： 你是一位经验丰富的医学专家。
                            # 任务：请根据以下患者信息，生成一个最准确的诊断结果。只需输出最终诊断名称，不要包含解释或其他内容。
                            # 要求：
                                1.只能基于输入的文本分析，不要加入自己的想法，不要编造不存在的症状和检验检查数据等
                                2.如无法判断该疾病的性质，如"细菌性"、"病毒性"时，不要加入此类限定词
                                3.如果无法就给定的内容做出诊断时，可按照主诉的症状描述做诊断，如主诉为"头晕数日"，可以诊断为"头晕"
                                4.仔细分析体格检查中的异常体征

                            # 患者信息：
                            - 性别：{gender}
                            - 年龄：{age}
                            - 主诉：{chief_complaint}
                            - 现病史：{present_illness}
                            - 既往史：{past_medical_history}
                            - 个人史：{personal_history}
                            - 体格检查：{physical_examination}

                            # 诊断：
                            """
        
        # 创建新的条目
        new_entry = {
            "instruction": instruction,
            "input": "",
            "output": output
        }
        
        processed_data.append(new_entry)
    
    # 保存处理后的数据到JSON文件
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))


    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    
    print(f"数据预处理完成！共转换 {len(processed_data)} 条记录。")
    print(f"转换后的数据已保存到: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # 设置输入和输出文件名
    input_file = "data/respiratory_medical_raw_data.json"  # 原始数据文件名
    output_file = "data/respiratory_medical_processed_data.json"  # 处理后的数据文件名
    
    # # 提示用户输入文件名（可选）
    # custom_input = input("请输入原始数据文件名(默认为 medical_raw_data.json): ")
    # if custom_input.strip():
    #     input_file = custom_input
    
    # custom_output = input("请输入输出文件名(默认为 medical_processed_data.json): ")
    # if custom_output.strip():
    #     output_file = custom_output
    
    # 执行数据预处理
    preprocess_medical_data(input_file, output_file)