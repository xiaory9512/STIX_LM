import dspy

GPT3 = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=2000)
colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=colbertv2, lm=GPT3)

import os
import json


class STIXDataset:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.data = self.load_data()

    def load_data(self):
        data = []
        files = os.listdir(self.folder_path)
        text_files = [f for f in files if f.endswith('.txt')]

        for text_file in text_files:
            base_name = os.path.splitext(text_file)[0]
            json_file = f"{base_name}.json"
            if json_file in files:
                with open(os.path.join(self.folder_path, text_file), 'r', encoding='utf-8') as tf:
                    question = tf.read()
                with open(os.path.join(self.folder_path, json_file), 'r', encoding='utf-8') as jf:
                    try:
                        answer = json.load(jf)
                    except json.JSONDecodeError:
                        continue
                data.append((question, answer, text_file))
        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


folder_path = os.path.join(os.getcwd(), 'data')
train_data = STIXDataset(folder_path)
print(len(train_data))
print(train_data)
print(train_data[0])

#train = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in train_data]
train = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer, _ in train_data]



class BasicQA(dspy.Signature):
    """Analyze the scenario to construct a graph in STIX JSON format with several objects."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="A list of valid json objects. Output must be like '[object1, object2...]'.Each object must have an unique ID.")
    # A list include valid json objects.
    #Start with single object. Each STIX object must has to have an ID.


# Assume dspy.Predict is correctly defined to use a model or some logic to generate answers
generate_answer = dspy.Predict(BasicQA)

pred = generate_answer(question=train[2].question)

import os
import json

output_dir = os.path.join(os.getcwd(), 'AI_gen')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_data = STIXDataset("D://LLM//STIX_official_example//examples//all_examples")
train = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer, _ in train_data]

for example, (_, _, original_text_file) in zip(train, train_data):
    pred = generate_answer(question=example.question)
    output_filename = original_text_file.replace('.txt', '_GEN.json')
    output_path = os.path.join(output_dir, output_filename)
    print(type(pred.answer))
    print(pred.answer)


    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pred.answer)

for example, (_, _, original_text_file) in zip(train, train_data):
    generate_answer_with_chain_of_thought = dspy.ChainOfThought(BasicQA)
    pred = generate_answer_with_chain_of_thought(question=example.question)
    output_filename = original_text_file.replace('.txt', '_COT.json')
    output_path1 = os.path.join(output_dir, output_filename)
    print(type(pred.answer))
    print(pred.answer)

    with open(output_path1, 'w', encoding='utf-8') as f:
        f.write(pred.answer)
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(pred.answer, f, ensure_ascii=False, indent=4)
# print(json.dumps(pred.answer, ensure_ascii=False, indent=4))




print("All answers have been processed and saved.")
