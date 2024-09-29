# 练习2: LLM心理咨询能力测试

### 1.任务介绍：

---

**作业名称：基于心理咨询资格考试题库的大语言模型心理咨询能力测试**

**任务目标：**

在本次作业中，你需要加载一个模型，来测试其对心理咨询资格考试题目的理解和处理能力。题库包含：

- 100道单项选择题

你的任务是使用代码实现一个程序，能够正确处理和回答这些题目。通过该项目，你将进一步理解自然语言处理中基于生成模型的分类问题，并且熟悉如何评估模型在选择题任务上的表现。

**具体要求：**

1. **题库加载**：从提供的题库中加载100道单项选择题。
2. **模型加载**：选择合适的模型来回答这些选择题。你可以使用API、自己部署或者训练一个模型来完成任务。
4. **模型评估**：评估模型在测试集上的表现，包括其对单项选择题和多项选择题的准确率。

---

### 评分标准：

| **评分项目**       | **分数** | **说明**                                                     |
| ------------------ | -------- | ------------------------------------------------------------ |
| **代码实现**       | 60%      | 能够正确加载题库、和实现合适的模型，对问题进行处理和分类。   |
| **结果分析与报告** | 30%      | 提供详细的结果分析，包括模型在不同类型题目上的表现差异，得出的准确率和误差分析等。 |
| **创新性**         | 10%      | 在模型设计或处理方式上有一定的创新性，能够体现出对问题的深入思考与探索。 |



### 2.代码实现

```python
import os
import re

import pandas as pd
import qianfan

os.environ["QIANFAN_AK"] = "XXX"  # 在这里填写API Key
os.environ["QIANFAN_SK"] = "XXX"  # 在这里填写Secret Key


def generate_answer(question: str):
    """
    用于调用大语言模型API生成对话回复
    """
    message_list = [{
        "role": "user",
        "content": question
    }]

    response_object = qianfan.ChatCompletion().do(  # 在下面可以修改参数，尝试不同的模型和效果
        model="ERNIE-4.0-8K", 
        messages=message_list, 
        temperature=0.95, 
        top_p=0.8, 
        penalty_score=1, 
        enable_system_memory=False, 
        disable_search=False, 
        enable_citation=False)
    
    response_msg = response_object.body["result"]

    return response_msg


def extract_ans(response_str):
        pattern=[
            r"^选([A-D])",
            r"^选项([A-D])",
            r"答案是\s?选?项?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?选?项?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
        ]
        ans_list=[]
        if response_str[0] in ["A",'B','C','D']:
            ans_list.append(response_str[0])
        for p in pattern:
            if len(ans_list)==0:
                ans_list=re.findall(p,response_str)
            else:
                break
        return ans_list


if __name__ == "__main__":
    single_choice_df = pd.read_csv("./data/single_choice_100.csv")

    single_choice_true, multi_choice_true = 0, 0

    for index, single_choice in single_choice_df.iterrows():
        question = single_choice["question"]
        for item in ["A", "B", "C", "D"]:
            question += f"\n{item}.{single_choice[item]}"
        question += "\n答案:"
        
        response = generate_answer(question)
        answer = extract_ans(response)

        if len(answer)>0 and (answer[-1]==single_choice["answer"]):
            single_choice_true += 1

    correct_ratio = 100 * single_choice_true / single_choice_df.size
```

