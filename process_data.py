import dspy
import os
import json

#latest version of model
#using different case to test few shot
#change the format of text input
#llama3.1 or Genmma 2
#gpt-3.5-turbo
#GPT3 = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=2500)
GPT4 = dspy.OpenAI(model='gpt-4-turbo', max_tokens=2500)
#llama3 = dspy.Llama(model='llama3')
colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
#colbertv2 = dspy.ColBERTv2(url='https://docs.oasis-open.org/cti/stix/v2.1/os/stix-v2.1-os.html#_k017w16zutw')
dspy.settings.configure(rm=colbertv2, lm=GPT4)


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

class OneShotRetriever(dspy.Retrieve):
    def __init__(self, example):
        super().__init__()
        self.example = example

    def forward(self, query):
        # Here we could use the query to determine if we should return the example
        # For demonstration, let's just print the query
        # print(f"Retrieval query: {query}")
        one_example = f"Example scenairo: {self.example.question}\n Example generated STIX in JSON based on the scenairo: {self.example.answer}\n"
        return one_example


folder_path = os.path.join(os.getcwd(), 'Process_data/4')
output_dir = folder_path
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
train_data = STIXDataset(folder_path)
print(len(train_data))
print(train_data)
#print(train_data[0])

#train = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in train_data]
train = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer, _ in train_data]
print("===")
print(train[0])

tune_data = STIXDataset(os.path.join(os.getcwd(), 'Process_data/one_shot_material'))
print(tune_data)
tune_materials = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer, _ in tune_data]
print(f"examples: {tune_materials}")
tune_material = tune_materials[0]



class BasicSITXGenerator(dspy.Signature):
    """Analyze the scenario to construct a graph in STIX JSON format with several objects."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="A list of valid json objects. Output must be like '[object1, object2...]'.Each object must have an unique ID. Each relationship must connect two evience entities in the list.")
    # A list include valid json objects.
    #Start with single object. Each STIX object must has to have an ID.

class SITXGeneratorSig(dspy.Signature):
    #"""Describe a conversation in STIX, which stands for Structured Threat Information eXpression, is a standardized language for representing cyber threat information."""
    """Analyze the scenario to construct a graph in STIX JSON format with several objects."""
    # Make sure to define context here, otherwise, one-short learning won't work
    context = dspy.InputField(desc="one example, which contains a scenario and the corresponding STIX in JSON")

    question: str = dspy.InputField(
        desc="a scenario describes a cyber crime"
    )

    answer: str = dspy.OutputField(
        desc="the formalized STIX in JSON representing cyber threat information based on the scenario, e.g., [{object 1}, {object 2}, ... {object n}]"
    )


class STIXPoTSig(dspy.Signature):
    """Analyze the scenario to construct a graph in STIX JSON with several objects."""
    context = dspy.InputField(desc="Contextual example for learning")
    question = dspy.InputField(desc="A scenario describing a cyber crime")
    answer = dspy.OutputField(desc="the formalized STIX in JSON representing cyber threat information based on the scenario, follow the format of [{object 1}, {object 2}, ... {object n}]")


class STXIGenCoT(dspy.Module):
    def __init__(self, example):
        super().__init__()
        self.retriever = OneShotRetriever(example)
        self.predictor = dspy.ChainOfThought(SITXGeneratorSig)

    def forward(self, question):
        context = self.retriever(question)
        results = self.predictor(context=context, question=question)

        return results


class STIXGenPoT(dspy.Module):
    def __init__(self, example):
        super().__init__()
        self.retriever = OneShotRetriever(example)
        self.predictor = dspy.ProgramOfThought(STIXPoTSig)

    def forward(self, question):
        context = self.retriever(question)
        results = self.predictor(context=context, question=question)
        return results


OneShotModule = STXIGenCoT(tune_material)
#train_data = STIXDataset("D://LLM//STIX_official_example//examples//all_examples")
#train_data = STIXDataset("D:\\ML\\dspy_dfkg_new\\Process_data\\data6")
#train = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer, _ in train_data]

for example, (_, _, original_text_file) in zip(train, train_data):
    generate_answer_Vanila_Predict = dspy.Predict(BasicSITXGenerator)
    pred_Valina_pred = generate_answer_Vanila_Predict(question=example.question)
    output_path_Valina_pred = os.path.join(output_dir, original_text_file.replace('.txt', '_Vanila_Predict.json'))
    print("==valina Predict==")
    with open(output_path_Valina_pred, 'w', encoding='utf-8') as f:
        f.write(pred_Valina_pred.answer)

    generate_answer_Vanila_COT = dspy.ChainOfThought(BasicSITXGenerator)
    #print(GPT4.inspect_history(n=2))
    pred_Vanila_COT = generate_answer_Vanila_COT(question=example.question)
    output_path_Vanila_COT = os.path.join(output_dir, original_text_file.replace('.txt', '_Vanila_COT.json'))
    #print(type(pred_Vanila_COT.answer))
    #print(pred_Vanila_COT.answer)
    print("==vanila COT==")
    with open(output_path_Vanila_COT, 'w', encoding='utf-8') as f:
        f.write(pred_Vanila_COT.answer)

    generate_answer_Oneshot_Predict = dspy.Predict(SITXGeneratorSig)
    pred_Oneshot_Predict = generate_answer_Oneshot_Predict(question=example.question)
    output_path_Oneshot_Predict = os.path.join(output_dir, original_text_file.replace('.txt', '_Oneshot_Predict.json'))
    # print(type(pred_Oneshot_COT.answer))
    print(pred_Oneshot_Predict.answer)
    print("==Oneshot Predict==")
    with open(output_path_Oneshot_Predict, 'w', encoding='utf-8') as f:
        f.write(pred_Oneshot_Predict.answer)


    generate_answer_Oneshot_COT = dspy.ChainOfThought(SITXGeneratorSig)
    pred_Oneshot_COT = OneShotModule(question=example.question)
    output_path_Oneshot_COT = os.path.join(output_dir, original_text_file.replace('.txt', '_Oneshot_COT.json'))
    #print(type(pred_Oneshot_COT.answer))
    print(pred_Oneshot_COT.answer)
    print("==Oneshot COT==")
    with open(output_path_Oneshot_COT, 'w', encoding='utf-8') as f:
        f.write(pred_Oneshot_COT.answer)

    generate_answer_Oneshot_POT = dspy.ProgramOfThought(STIXPoTSig)
    pred_Oneshot_POT = OneShotModule(question=example.question)
    output_path_Oneshot_POT = os.path.join(output_dir, original_text_file.replace('.txt', '_Oneshot_POT.json'))
    print(pred_Oneshot_POT.answer)
    print("==Oneshot POT==")
    with open(output_path_Oneshot_POT, 'w', encoding='utf-8') as f:
        f.write(pred_Oneshot_POT.answer)

    # #Zero shot Program of thoughts
    # generate_answer_Vanila_POT= dspy.ProgramOfThought(BasicSITXGenerator)
    # pred_Vanila_POT = generate_answer_Vanila_POT(question=example.question)
    # output_path_Vanila_POT = os.path.join(output_dir, original_text_file.replace('.txt', '_Vanila_POT.json'))
    # # print(type(pred_Vanila_COT.answer))
    # # print(pred_Vanila_COT.answer)
    # print("==vanila POT==")
    # with open(output_path_Vanila_POT, 'w', encoding='utf-8') as f:
    #     f.write(pred_Vanila_POT.answer)
    #
    # #One shot Porgram of thoughts
    # generate_answer_Oneshot_POT = dspy.ProgramOfThought(SITXGeneratorSig)
    # pred_Oneshot_POT = OneShotModule(question=example.question)
    # output_path_Oneshot_POT = os.path.join(output_dir, original_text_file.replace('.txt', '_Oneshot_POT.json'))
    # # print(type(pred_Oneshot_COT.answer))
    # print(pred_Oneshot_POT.answer)
    # print("==Oneshot POT==")
    # with open(output_path_Oneshot_POT, 'w', encoding='utf-8') as f:
    #     f.write(pred_Oneshot_POT.answer)




print("All answers have been processed and saved.")
