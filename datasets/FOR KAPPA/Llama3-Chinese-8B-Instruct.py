import torch
from modelscope import snapshot_download
from modelscope import AutoModelForCausalLM
from transformers import AutoTokenizer
import csv
import transformers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# 下载并加载模型
model_dir = snapshot_download('FlagAlpha/Llama3-Chinese-8B-Instruct')
pipeline = transformers.pipeline(
    "text-generation",
    model=model_dir,
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda",
)

# 定义提示模板
# prompt_template = (
#     "你是一个智能电商直播助手，负责判断用户弹幕是否表现出购买意图。"
#     "购买意图指的是用户在弹幕中表达对商品的兴趣或对购买商品的倾向，例如使用场景关联、询问价格、积极评价、决策辅助请求或讨论商品特性等。"
#     "如果弹幕仅是售后、物流、差评、闲聊、主播个人提问、陈述感受或与商品无关的气氛词，则无购买意图。\n"
#     "请你根据下面给你的弹幕内容判断该弹幕是否具有商品推荐价值，从而触发商品推荐逻辑\n"
#     "注意：最后输出的结果格式是1个0或1的数字（0代表当前弹幕没有商品推荐价值，1代表当前弹幕有商品推荐价值），不用解释。\n"
# )

prompt_template = (
    "你是一个智能电商直播助手，负责判断用户弹幕是否表现出购买意图。"
    "注意：最后输出的结果格式是1个0或1的数字（0代表没有，1代表有），不用解释。\n"
)

def process_csv(input_csv, output_csv):
    true_labels = []
    pred_labels = []
    processed_count = 0

    with open(input_csv, 'r', encoding='utf-8', errors='ignore') as infile, \
            open(output_csv, 'w', encoding='utf-8', newline='') as outfile:

        csv_reader = csv.reader(infile)
        csv_writer = csv.writer(outfile)

        # 读取并写入标题行
        headers = next(csv_reader)
        headers.append('预测标签')  # 添加预测标签列
        csv_writer.writerow(headers)

        for row in csv_reader:
            if len(row) < 5:  # 确保行有足够列
                continue

            danmu = row[2]
            label = row[3].strip()

            # 跳过无效标签
            if label not in ['0', '1']:
                continue

            true_labels.append(int(label))

            # 准备输入
            content = prompt_template + danmu
            messages = [{"role": "user", "content": content}]

            try:
                prompt = pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                terminators = [
                    pipeline.tokenizer.eos_token_id,
                    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                outputs = pipeline(
                    prompt,
                    max_new_tokens=10,  # 减少token数量，因为我们只需要0/1
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9
                )

                response = outputs[0]["generated_text"][len(prompt):].strip()
                print("="*50)
                print("user:",content)
                print("=" * 50)
                print("LLM:",response)
                # 从响应中提取0/1
                pred = 0
                if '1' in response:
                    pred = 1

                pred_labels.append(pred)

                # 将预测结果添加到原始行并写入新文件
                new_row = row.copy()
                new_row.append(str(pred))
                csv_writer.writerow(new_row)

                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"已处理 {processed_count} 条弹幕...")

            except Exception as e:
                print(f"处理弹幕时出错: {danmu}, 错误: {e}")
                # 出错时默认预测为0并写入
                new_row = row.copy()
                new_row.append('0')
                csv_writer.writerow(new_row)
                pred_labels.append(0)

    return true_labels, pred_labels


# 输入输出文件路径
input_csv = '四个平台数据集/dy-test.csv'
output_csv = '四个平台数据集/dy-test-result.csv'

# 检查输入文件是否存在
if not os.path.exists(input_csv):
    print(f"错误: 输入文件 {input_csv} 不存在!")
    exit()

print("开始处理弹幕数据...")
true_labels, pred_labels = process_csv(input_csv, output_csv)
print(f"处理完成! 结果已保存到 {output_csv}")

# 计算评估指标
if len(true_labels) > 0 and len(pred_labels) > 0:
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels,average='macro')
    recall = recall_score(true_labels, pred_labels,average='macro')
    f1 = f1_score(true_labels, pred_labels,average='macro')

    print("\n评估结果:")
    print(f"总样本数: {len(true_labels)}")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-score): {f1:.4f}")

    # 将评估结果也写入文件
    with open(output_csv, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(['评估指标', '值'])
        writer.writerow(['总样本数', len(true_labels)])
        writer.writerow(['准确率', f"{accuracy:.4f}"])
        writer.writerow(['精确率', f"{precision:.4f}"])
        writer.writerow(['召回率', f"{recall:.4f}"])
        writer.writerow(['F1分数', f"{f1:.4f}"])
else:
    print("没有有效数据可评估")