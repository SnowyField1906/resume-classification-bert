# Báo cáo đồ án Learning Statistic ( Học thống kê)

### Báo cáo Đồ án: Phân loại Văn bản và Phân tích CV

### Mục lục

I - Phần xử lý dữ liệu: 

1. Giới thiệu
2. Bài toán Phân loại Văn bản
3. Bài toán Phân tích CV

II - Xây dựng ứng dụng: 

a - Cấu trúc folder

---

## I - Phần xử lý dữ liệu:

### 1. Giới thiệu

Đồ án này tập trung vào việc áp dụng mô hình học sâu dựa trên kiến trúc Transformer, cụ thể là BERT, để giải quyết hai bài toán: phân loại văn bản và phân tích CV. Mục tiêu chính của đồ án là xây dựng một mô hình có khả năng phân loại các loại văn bản khác nhau và một mô hình phân tích CV nhằm trích xuất và phân tích thông tin từ các bản CV.

### 2. Bài toán Phân loại Văn bản ( Text Classification )

Phân loại văn bản là một bài toán trong lĩnh vực Xử lý Ngôn ngữ Tự nhiên (NLP), nhằm xác định chủ đề hoặc phân loại văn bản vào các danh mục khác nhau. Việc phân loại văn bản có thể ứng dụng trong nhiều lĩnh vực như lọc email, phân tích cảm xúc, và phân loại tin tức.

**Vấn đề liên quan:**

- Xử lý dữ liệu văn bản không có cấu trúc.
- Đảm bảo độ chính xác cao khi phân loại.

**Tầm quan trọng:**

- Tự động hóa quy trình phân loại văn bản giúp tiết kiệm thời gian và nguồn lực.
- Tăng độ chính xác và hiệu quả trong các ứng dụng thực tế.

### 3. Vận dụng Text- Classification vào bài toán Phân tích CV

Phân tích CV là quá trình tự động trích xuất thông tin từ các bản CV, giúp các nhà tuyển dụng hoặc hệ thống tuyển dụng tự động xác định được ngành nghề thích hợp của ứng viên thông qua các kỹ năng, kinh nghiệm và thông tin liên quan được để cập đến trong CV.

**Cách thức hoạt động:**

- Trích xuất thông tin từ văn bản không cấu trúc.
- Sử dụng các mô hình NLP để nhận dạng các thực thể quan trọng như tên, kỹ năng, kinh nghiệm làm việc, v.v.

**Tầm quan trọng:**

- Tự động hóa quy trình tuyển dụng giúp tăng tốc độ và hiệu quả.
- Hỗ trợ tìm kiếm công việc hiệu quả, phù hợp với năng lực.
- Giảm thiểu sai sót và bias trong quá trình sàng lọc hồ sơ.

### GPT vs BERT:

GPT (Generative Pre-trained Transformer) và BERT (Bidirectional Encoder Representations from Transformers) đều là hai mô hình nổi bật trong lĩnh vực xử lý ngôn ngữ tự nhiên, nhưng chúng có những đặc tính và ứng dụng khác nhau.

### Đặc điểm chung:

1. **Kiến trúc Transformer:** Cả GPT và BERT đều dựa trên kiến trúc Transformer, là một trong những phương pháp hiệu quả nhất hiện nay cho việc xử lý ngôn ngữ tự nhiên.
2. **Đã được huấn luyện trước:** Cả hai đều là các mô hình đã được huấn luyện trước trên lượng dữ liệu lớn, có khả năng học được các biểu diễn ngôn ngữ phong phú.

### Sự khác biệt:

1. **Mục đích huấn luyện:**
    - **GPT:** Được huấn luyện để sinh ra văn bản mới từ ngữ cảnh đã cho.
    - **BERT:** Được huấn luyện để hiểu và dự đoán các từ điển hình trong văn bản một cách đối xứng, có tính hai chiều.
2. **Ứng dụng trong bài toán phân loại CV:**
    - **BERT:** Thường được ưa chuộng hơn trong các bài toán đặt ra câu hỏi, phân loại văn bản, hoặc làm việc với các đoạn văn ngắn. Với BERT, bạn có thể sử dụng ngữ cảnh bao quát để hiểu và phân loại các mẫu CV. BERT có khả năng xử lý ngữ nghĩa của câu văn tốt hơn nhờ vào đặc trưng bidirectional của nó, cho phép nó hiểu bối cảnh xung quanh các từ và cụm từ trong một câu.
    - **GPT:** Thường được sử dụng để sinh ra văn bản mới hoặc hoàn thiện các đoạn văn dài hơn. GPT có xu hướng sinh ra văn bản trơn hơn và phong phú hơn nhờ vào mục đích huấn luyện và kiến trúc mà nó đi theo.

### Lựa chọn BERT cho bài toán phân loại CV:

- **Khả năng hiểu ngữ nghĩa câu:** BERT có khả năng hiểu ngữ nghĩa của các câu văn và ngữ cảnh xung quanh từ hoặc cụm từ, điều này hữu ích khi cần phân loại CV dựa trên mô tả công việc và kinh nghiệm.
- **Phân loại đa nhãn:** BERT có thể được điều chỉnh để phân loại đa nhãn, nghĩa là có thể gắn nhãn cho một CV vào nhiều role công việc khác nhau nếu cần.
- **Sử dụng pre-trained model:** BERT đã được huấn luyện trên dữ liệu lớn và có thể được fine-tuning trên tập dữ liệu CV cụ thể của bạn để cải thiện hiệu suất phân loại

**Các thuật toán và công nghệ sử dụng:**

- **BERT (Bidirectional Encoder Representations from Transformers):** Sử dụng BERT để trích xuất các đặc trưng ngữ nghĩa từ văn bản.
- **Fine-tuning:** Huấn luyện lại mô hình BERT trên các tập dữ liệu cụ thể để tăng độ chính xác.
- **Tokenization:** Sử dụng tokenization để chuyển đổi văn bản thành các token mà mô hình có thể xử lý.

**Công cụ và môi trường:**

- **Python:** Ngôn ngữ lập trình chính.
- **PyTorch:** Thư viện học sâu để xây dựng và huấn luyện mô hình.
- **Hugging Face Transformers:** Thư viện hỗ trợ các mô hình transformer.

### **Chi tiết về code và các bước thực hiện**

1. **Chuẩn bị dữ liệu:** Đọc và xử lý các tập dữ liệu đầu vào.
2. Exploring Data Analysis (EDA)
3. **Tiền xử lý dữ liệu:**
4. **Huấn luyện mô hình:** Sử dụng tập dữ liệu đã chuẩn bị để huấn luyện mô hình BERT.
5. **Đánh giá mô hình:** Sử dụng các độ đo như độ chính xác, F1-score để đánh giá.
6. **Xây dựng ứng dụng:** Triển khai mô hình vào một ứng dụng web để thực hiện phân loại văn bản và phân tích CV tự động.

### Chuẩn bị dữ liệu ( Data preparation )

```python
import pandas as pd

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data.csv')

# Xem xét một số dòng trong dữ liệu
print(data.head())

```

### Phân tích khám phá dữ liệu (EDA):

```python
sns.countplot(df.Category)
plt.xticks(rotation=90)
plt.tight_layout();
```

![image](https://github.com/SnowyField1906/resume-classification-bert/assets/57946382/d8e32e41-6fc8-4797-96fe-5bc45b5bc716)


```python
print("Total number of available jobs:", df.Category.nunique())
```

→ Total number of available jobs: 25

```python
df['resume_len'] = df.Resume.apply(len)
sns.distplot(df.resume_len);
```

![Untitled](Ba%CC%81o%20ca%CC%81o%20%C4%91o%CC%82%CC%80%20a%CC%81n%20Learning%20Statistic%20(%20Ho%CC%A3c%20tho%CC%82%CC%81%201b2488529fde42b286596a8ebf2e552b/Untitled%202.png)

```python
df.describe()
```

![Untitled](Ba%CC%81o%20ca%CC%81o%20%C4%91o%CC%82%CC%80%20a%CC%81n%20Learning%20Statistic%20(%20Ho%CC%A3c%20tho%CC%82%CC%81%201b2488529fde42b286596a8ebf2e552b/Untitled%203.png)

```python
max_resume_len = 200
```

```python
wc = WordCloud(width=600,height=300,random_state=101).generate(' '.join(df.Resume))
plt.title(label='Word Cloud of Resume Sections',pad=20,fontsize=25,fontweight='bold',color='sienna')
plt.imshow(wc);
```

![Untitled](Ba%CC%81o%20ca%CC%81o%20%C4%91o%CC%82%CC%80%20a%CC%81n%20Learning%20Statistic%20(%20Ho%CC%A3c%20tho%CC%82%CC%81%201b2488529fde42b286596a8ebf2e552b/Untitled%204.png)

Gắn nhãn với các công việc có sẵn

```jsx
labels_dict = {}

for idx, label in enumerate(df.Category.unique()):
    labels_dict[label] = idx

labels_dict
```

→ 

{'Data Science': 0,
'HR': 1,
'Advocate': 2,
'Arts': 3,
'Web Designing': 4,
'Mechanical Engineer': 5,
'Sales': 6,
'Health and fitness': 7,
'Civil Engineer': 8,
'Java Developer': 9,
'Business Analyst': 10,
'SAP Developer': 11,
'Automation Testing': 12,
'Electrical Engineering': 13,
'Operations Manager': 14,
'Python Developer': 15,
'DevOps Engineer': 16,
'Network Security Engineer': 17,
'PMO': 18,
'Database': 19,
'Hadoop': 20,
'ETL Developer': 21,
'DotNet Developer': 22,
'Blockchain': 23,
'Testing': 24}

```jsx
df.Category = df.Category.apply(func=lambda x: labels_dict[x])
df.Category = df.Category.astype(np.int64)
```

### Tiền xử lý dữ liệu ( Data preprocessing ):

Hàm `text_preprocess` được thiết kế để tiền xử lý dữ liệu văn bản trong một cột của dataframe. Các bước tiền xử lý bao gồm loại bỏ HTML tags, loại bỏ stop words, các con số, các liên kết, các ký tự đặc biệt, dấu câu, ký tự không phải ASCII, địa chỉ email và chuyển đổi văn bản thành chữ thường. Mỗi bước này nhằm chuẩn hóa và làm sạch dữ liệu để thuận tiện cho các công đoạn xử lý và phân tích văn bản tiếp theo.

```python
def text_preprocess(data,col):
    data[col] = data[col].apply(func=clean_html)
    data[col] = data[col].apply(func=remove_)
    data[col] = data[col].apply(func=removeStopWords)
    data[col] = data[col].apply(func=remove_digits)
    data[col] = data[col].apply(func=remove_links)
    data[col] = data[col].apply(func=remove_special_characters)
    data[col] = data[col].apply(func=punct)
    data[col] = data[col].apply(func=non_ascii)
    data[col] = data[col].apply(func=email_address)
    data[col] = data[col].apply(func=lower)
    return data
    
preprocessed_df = text_preprocess(df,'Resume')
preprocessed_df.head()

```

![Untitled](Ba%CC%81o%20ca%CC%81o%20%C4%91o%CC%82%CC%80%20a%CC%81n%20Learning%20Statistic%20(%20Ho%CC%A3c%20tho%CC%82%CC%81%201b2488529fde42b286596a8ebf2e552b/Untitled%205.png)

### **Tải xuống pre-trained tokenizer và DistilBert model:**

Chúng em sử dụng từ model có sẵn “manishiitg/distilbert-resume-parts-classify” ([https://huggingface.co/manishiitg/distilbert-resume-parts-classify](https://huggingface.co/manishiitg/distilbert-resume-parts-classify)) và tinh chỉnh lại từ model trên.

```jsx
tokenizer = AutoTokenizer.from_pretrained("manishiitg/distilbert-resume-parts-classify")
bert_model = TFDistilBertForSequenceClassification.from_pretrained("manishiitg/distilbert-resume-parts-classify",from_pt=True)
```

### Chia tập dữ liệu đã được xử lý trước thành tập huấn luyện và tập kiểm tra:

```jsx
train_df, test_df = train_test_split(preprocessed_df,test_size=0.3,shuffle=True,random_state=101)

```

### Text tokenization:

```jsx
X_train = tokenizer(text=train_df.Resume.tolist(),
                   add_special_tokens=True,
                   padding=True,
                   truncation=True,
                   max_length=max_resume_len,
                   return_tensors='tf',
                   return_attention_mask=True,
                   return_token_type_ids=False,
                   verbose=1)

X_test = tokenizer(text=test_df.Resume.tolist(),
                  add_special_tokens=True,
                  padding=True,
                  truncation=True,
                  max_length=max_resume_len,
                  return_tensors='tf',
                  return_attention_mask=True,
                  return_token_type_ids=False,
                  verbose=1)
```

### Xác định kiến trúc mô hình:

Mô hình này tổng cộng được gắn thêm 13 lớp, bao gồm các lớp biến đổi embedding từ BERT, các lớp Dense để học đặc trưng, các lớp chuẩn hóa và Dropout để điều chỉnh và ngăn chặn overfitting, và lớp đầu ra để dự đoán phân loại vào 25 nhãn khác nhau là các  role  trong dataset.  Mỗi lớp được thiết kế để học một cách hợp lý các đặc trưng từ dữ liệu và chuẩn bị dữ liệu để đưa ra dự đoán chính xác.

```jsx
input_ids = Input(shape=(max_resume_len,),dtype=tf.int32,name='input_ids')
attention_masks = Input(shape=(max_resume_len,),dtype=tf.int32,name='attention_mask')
word_embeddings = bert_model(input_ids,attention_mask=attention_masks)[0] # 0 --> final hidden state, 1 --> pooling output

output = Flatten()(word_embeddings)
output = Dense(units=1024,activation='relu')(output)
output = BatchNormalization()(output)
output = Dropout(0.25)(output)
output = Dense(units=512,activation='relu')(output)
output = Dropout(0.25)(output)
output = Dense(units=256,activation='relu')(output)
output = BatchNormalization()(output)
output = Dropout(0.25)(output)
output = Dense(units=128,activation='relu')(output)
output = Dropout(0.25)(output)
output = Dense(units=64,activation='relu')(output)
output = Dense(units=25,activation='softmax')(output)

model = Model(inputs=[input_ids,attention_masks],outputs=output)
model.layers
```

### Minh họa kiến trúc mô hình:

```jsx
model.summary()
```

![Untitled](Ba%CC%81o%20ca%CC%81o%20%C4%91o%CC%82%CC%80%20a%CC%81n%20Learning%20Statistic%20(%20Ho%CC%A3c%20tho%CC%82%CC%81%201b2488529fde42b286596a8ebf2e552b/Untitled%206.png)

### Huấn luyện mô hình

```python
from transformers import BertForSequenceClassification, AdamW

# Load model BERT cho nhiệm vụ phân loại
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Huấn luyện mô hình
model.train()
for epoch in range(3):
    outputs = model(input_ids, labels=torch.tensor(data['label'].values))
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

```

```jsx
plot_model(model,to_file='model.png',dpi=100,show_shapes=True)
```

Plot : 

![Untitled](Ba%CC%81o%20ca%CC%81o%20%C4%91o%CC%82%CC%80%20a%CC%81n%20Learning%20Statistic%20(%20Ho%CC%A3c%20tho%CC%82%CC%81%201b2488529fde42b286596a8ebf2e552b/Untitled%207.png)

### Biên dịch mô hình:

```jsx
adam = Adam(learning_rate=5e-5,
           epsilon=2e-8,
           decay=0.01,
           clipnorm=1.0)

model.compile(loss='sparse_categorical_crossentropy',optimizer=adam,metrics=SparseCategoricalAccuracy('balanced_accuracy'))
```

### Huấn luyện mô hình Distil-Bert đã được tinh chỉnh:

```jsx
es = EarlyStopping(monitor='val_balanced_accuracy',patience=250,verbose=1,mode='max',restore_best_weights=True)
mc = ModelCheckpoint('resume_parser.h5',monitor='val_balanced_accuracy',mode='max',verbose=1,save_best_only=True)

r = model.fit(x={'input_ids': X_train['input_ids'], 'attention_mask': X_train['attention_mask']},
             y=train_df.Category,
             epochs=500,
             batch_size=32,
             callbacks=[es,mc],
             validation_data=({'input_ids': X_test['input_ids'], 'attention_mask': X_test['attention_mask']},test_df.Category))
```

### Minh họa hiệu xuất mô hình:

```jsx
plt.plot(r.history['loss'],'r',label='train loss')
plt.plot(r.history['val_loss'],'b',label='test loss')
plt.xlabel('No. of Epochs')
plt.ylabel('Categorical Crossentropy Loss')
plt.title('Loss Graph')
plt.legend();
```

![Untitled](Ba%CC%81o%20ca%CC%81o%20%C4%91o%CC%82%CC%80%20a%CC%81n%20Learning%20Statistic%20(%20Ho%CC%A3c%20tho%CC%82%CC%81%201b2488529fde42b286596a8ebf2e552b/Untitled%208.png)

```jsx
plt.plot(r.history['balanced_accuracy'],'r',label='train accuracy')
plt.plot(r.history['val_balanced_accuracy'],'b',label='test accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Balanced Categorical Accuracy')
plt.title('Accuracy Graph')
```

![Untitled](Ba%CC%81o%20ca%CC%81o%20%C4%91o%CC%82%CC%80%20a%CC%81n%20Learning%20Statistic%20(%20Ho%CC%A3c%20tho%CC%82%CC%81%201b2488529fde42b286596a8ebf2e552b/Untitled%209.png)

### Đánh giá hiệu xuất mô hình:

```jsx
loss, acc = model.evaluate({'input_ids': X_test['input_ids'], 'attention_mask': X_test['attention_mask']},test_df.Category)
print("Test Sparse Categorical Crossentropy Loss:", loss)
print("Test Balanced Categorical Accuracy:", acc)
```

10/10 [==============================] - 2s 190ms/step - loss: 0.1330 - balanced_accuracy: 0.9931
Test Sparse Categorical Crossentropy Loss: 0.13300560414791107
Test Balanced Categorical Accuracy: 0.9930796027183533

```jsx
test_predictions = model.predict({'input_ids': X_test['input_ids'], 'attention_mask': X_test['attention_mask']})
test_predictions = np.argmax(test_predictions,axis=1)
test_predictions
```

array([ 3, 14,  0,  0, 19, 10, 24, 21, 19, 15, 18, 13, 24, 24, 14,  7,  3,
17, 23,  0, 13, 17,  9, 11, 14, 20,  4,  9, 17,  9, 20,  4,  4, 21,
6, 12,  9, 23, 12, 24, 14, 24, 10,  9,  9, 18, 10, 21, 23,  3, 14,
20, 14, 12,  1, 24,  6, 23, 16,  6, 12, 21, 14, 23, 24,  0, 16,  5,
7, 22, 10,  0,  6,  2, 23,  3,  8,  3, 15, 18, 24, 19, 17,  3,  4,
22, 11, 20, 14, 24,  9, 21,  0, 15,  3, 20, 20,  4, 10,  9,  9, 24,
17, 11, 16,  1,  5,  3,  3,  9, 23, 17, 16,  0, 18, 22, 20,  4, 14,
16, 14, 21, 11,  3,  5, 20,  1, 23,  5,  5,  8, 22,  6, 10, 15,  9,
3,  9, 18,  0, 19, 10,  6,  5,  9,  7, 19,  5,  6,  4,  3,  4, 11,
6,  9,  1,  5, 12, 20, 17, 13,  4, 13,  9, 16, 15,  9,  0, 24, 21,
24, 21, 10, 18, 11, 11, 15,  5, 21, 24,  0,  4, 20, 18, 21, 24, 14,
4,  6,  4,  0,  9, 21, 13,  4, 21,  4, 11, 14,  4,  5,  7, 17,  1,
6,  7,  9, 23, 18, 14,  0, 23, 12, 23,  3, 21,  9, 21,  3, 20,  1,
20, 14, 16, 20, 21, 15, 20, 21,  4, 24, 21, 18, 14, 12, 21, 22,  1,
9, 19,  5, 22,  5, 16, 20, 19,  9,  8,  9,  3, 17, 15, 18, 17, 18,
9,  9, 14,  5, 12,  3, 24,  0, 23, 16,  7, 12, 15,  7, 11,  4, 20,
22, 19,  7,  4, 24,  3,  6,  4, 21, 16,  8,  9, 15, 15, 19, 18,  2])

```jsx
test_predictions
print("Confusion Matrix:")
print(confusion_matrix(test_df.Category,test_predictions))
print("Classification Report:")
print(classification_report(test_df.Category,test_predictions))
```

Confusion Matrix:
[[13  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0]
[ 0  7  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0]
[ 0  0  2  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0
0]
[ 0  0  0 17  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0]
[ 0  0  0  0 19  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0]
[ 0  0  0  0  0 13  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0]
[ 0  0  0  0  0  0 11  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0]
[ 0  0  0  0  0  0  0  8  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0]
[ 0  0  0  0  0  0  0  0  4  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0]
[ 0  0  0  0  0  0  0  0  0 25  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0]
[ 0  0  0  0  0  0  0  0  0  0  8  0  0  0  0  0  0  0  0  0  0  0  0  0
0]
[ 0  0  0  0  0  0  0  0  0  0  0  7  0  0  0  0  0  0  0  0  0  0  0  0
0]
[ 0  0  0  0  0  0  0  0  0  0  0  0  9  0  0  0  0  0  0  0  0  0  0  0
0]
[ 0  0  0  0  0  0  0  0  0  0  0  0  0  5  0  0  0  0  0  0  0  0  0  0
0]
[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 16  0  0  0  0  0  0  0  0  0
0]
[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 11  0  0  0  0  0  0  0  0
0]
[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0  0  0  0  0
0]
[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0  0  0  0
0]
[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 12  0  0  0  0  0
0]
[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  9  0  0  0  0
0]
[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 16  0  0  0
0]
[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 19  0  0
0]
[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  7  0
0]
[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 12
0]
[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
17]]
Classification Report:
precision    recall  f1-score   support

 0       1.00      1.00      1.00        13
       1       1.00      1.00      1.00         7
       2       1.00      0.50      0.67         4
       3       1.00      1.00      1.00        17
       4       1.00      1.00      1.00        19
       5       1.00      1.00      1.00        13
       6       1.00      1.00      1.00        11
       7       1.00      1.00      1.00         8
       8       1.00      1.00      1.00         4
       9       1.00      1.00      1.00        25
      10       1.00      1.00      1.00         8
      11       0.78      1.00      0.88         7
      12       1.00      1.00      1.00         9
      13       1.00      1.00      1.00         5
      14       1.00      1.00      1.00        16
      15       1.00      1.00      1.00        11
      16       1.00      1.00      1.00        10
      17       1.00      1.00      1.00        10
      18       1.00      1.00      1.00        12
      19       1.00      1.00      1.00         9
      20       1.00      1.00      1.00        16
      21       1.00      1.00      1.00        19
      22       1.00      1.00      1.00         7
      23       1.00      1.00      1.00        12
      24       1.00      1.00      1.00        17

accuracy                           0.99       289

macro avg       0.99      0.98      0.98       289
weighted avg       0.99      0.99      0.99       289

### 5. Kết quả thực nghiệm

Dưới đây là kết quả thực nghiệm của chúng em, bao gồm các số liệu và biểu đồ minh họa.

**Phân loại Văn bản:**

- **Độ chính xác:** xx%
- **F1-score:** xx%
- **Biểu đồ Confusion Matrix:** Cho thấy các danh mục văn bản được phân loại chính xác.

**Phân tích CV:**

- **Độ chính xác:** xx%
- **Precision:xx**%
- **Recall:** xx%
- **Biểu đồ phân tích thông tin trích xuất:** Cho thấy các loại thông tin như kỹ năng, kinh nghiệm được trích xuất với độ chính xác cao.

### 6. Kết luận và Hướng phát triển

Đồ án đã thành công trong việc xây dựng các mô hình phân loại văn bản và phân tích CV với độ chính xác cao. Tuy nhiên, còn nhiều hướng phát triển để cải thiện:

**Hướng phát triển:**

- **Mở rộng dữ liệu huấn luyện:** Sử dụng nhiều dữ liệu hơn để tăng độ chính xác.
- **Cải thiện mô hình:** Thử nghiệm với các mô hình tiên tiến hơn như GPT-3 hoặc các biến thể của BERT.
- **Ứng dụng thực tế:** Triển khai mô hình vào các hệ thống thực tế để kiểm tra và cải thiện hiệu suất.

---

## II - Xây dựng ứng dụng dựa trên mô hình đã huấn luyện:
### a - Cấu trúc folder:

