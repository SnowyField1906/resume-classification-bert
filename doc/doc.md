# Báo cáo đồ án Learning Statistic ( Học thống kê)

## Đề tài: Phân loại Văn bản và Phân tích CV

### Mục lục

[I - Thành viên nhóm:](#thanhvien)

[II - Phần xử lý dữ liệu:](#xulydulieu)

[1. Giới thiệu](#gioithieu)

[2. Bài toán Phân loại Văn bản](#baitoanphanloai)

[3. Bài toán Phân tích CV](#baitoanphantichcv)

[III - Xây dựng ứng dụng:](#xaydungungdung) 

[1. Cấu trúc folder](#cautrucfolder)

[2. Setup môi trường và chạy chương trình](#setupandrun)

[IV - Tham khảo:](#thamkhao)


---

## I- Thành viên nhóm:
<a name="thanhvien"></a>


xxxxxxxx - Nguyễn Hữu Thuận

xxxxxxxx - Thạch Thông

xxxxxxxx - Võ Gia Khang

21120090 - Mai Trần Phú Khương 


## II - Phần xử lý dữ liệu:
<a name="xulydulieu"></a>


### 1. Giới thiệu
<a name="gioithieu"></a>

Đồ án này tập trung vào việc áp dụng mô hình học sâu dựa trên kiến trúc Transformer, cụ thể là BERT, để giải quyết hai bài toán: phân loại văn bản và phân tích CV. Mục tiêu chính của đồ án là xây dựng một mô hình có khả năng phân loại các loại văn bản khác nhau và một mô hình phân tích CV nhằm trích xuất và phân tích thông tin từ các bản CV.

### 2. Bài toán Phân loại Văn bản ( Text Classification )
<a name="baitoanphanloai"></a>


Phân loại văn bản là một bài toán trong lĩnh vực Xử lý Ngôn ngữ Tự nhiên (NLP), nhằm xác định chủ đề hoặc phân loại văn bản vào các danh mục khác nhau. Việc phân loại văn bản có thể ứng dụng trong nhiều lĩnh vực như lọc email, phân tích cảm xúc, và phân loại tin tức.

**Vấn đề liên quan:**

- Xử lý dữ liệu văn bản không có cấu trúc.
- Đảm bảo độ chính xác cao khi phân loại.

**Tầm quan trọng:**

- Tự động hóa quy trình phân loại văn bản giúp tiết kiệm thời gian và nguồn lực.
- Tăng độ chính xác và hiệu quả trong các ứng dụng thực tế.

### 3. Vận dụng Text- Classification vào bài toán Phân tích CV
<a name="baitoanphantichcv"></a>

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

1. **Chuẩn bị dữ liệu và Tiền xử lí dữ liệu:** Đọc và xử lý các tập dữ liệu đầu vào.
4. **Huấn luyện mô hình:** Sử dụng tập dữ liệu đã chuẩn bị để huấn luyện mô hình BERT.
5. **Đánh giá mô hình:** Sử dụng các độ đo như độ chính xác, F1-score để đánh giá.

### 1. Chuẩn bị dữ liệu và tiền xử lí dữ liệu ( Data preparation & Preprocessing )
### Lớp DataFrame:
Cung cấp các phương thức để đổi tên cột, tính toán thêm thông tin cho dữ liệu, tiền xử lý văn bản và chuyển đổi nhãn

- Hàm '__prepocess__': thực hiện tiền xử lý dữ liệu thông qua lớp TextProccessor
- Hàm '__labels__': truy xuất thông tin của các labels sau khi train.

```python
class DataFrame:
    def __init__(self, path: str, x_col: str, y_col: str):
        ...

    def preprocess(self):
        ...

    def labels(self, output: list[int]) -> dict[str, str]:        
        ...

        return labels
```

- Hàm khởi tạo '__init__': thực hiện các thao tác xử lí nhãn từ dữ liệu, gán nhãn cho các công việc có sẵn

``` python
      data = pd.read_csv(path)
        data.rename(columns={x_col: "x", y_col: "y"}, inplace=True)

        data["resume_len"] = data.x.apply(len

        labels_dict = {}
        for idx, label in enumerate(data.y.unique()):
            labels_dict[label] = idx

        data.y = data.y.apply(func=lambda x: labels_dict[x])
        data.y = data.y.astype(np.int64)

        self.data = data
        self.labels_dict = labels_dict
```

### Gắn nhãn cho các công việc có sẵn:

→ {'Data Science': 0,
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


### Biểu đồ phân phối nhãn:

```python
sns.countplot(df.Category)
plt.xticks(rotation=90)
plt.tight_layout();
```

![image](https://github.com/SnowyField1906/resume-classification-bert/assets/57946382/d8e32e41-6fc8-4797-96fe-5bc45b5bc716)

```jsx
df.Category = df.Category.apply(func=lambda x: labels_dict[x])
df.Category = df.Category.astype(np.int64)
```

### Tiền xử lý dữ liệu ( Data preprocessing ):
### Lớp TextPreproccessor:

Lớp `TextPreprocessor` được thiết kế để tiền xử lý dữ liệu văn bản trong một cột của dataframe. Các bước tiền xử lý bao gồm loại bỏ HTML tags, loại bỏ stop words, các con số, các liên kết, các ký tự đặc biệt, dấu câu, ký tự không phải ASCII, địa chỉ email và chuyển đổi văn bản thành chữ thường. Mỗi bước này nhằm chuẩn hóa và làm sạch dữ liệu để thuận tiện cho các công đoạn xử lý và phân tích văn bản tiếp theo.

```python
class TextPreprocessor:
    def __init__(self):
        self.cached_stop_words = set(stopwords.words("english"))
        self.cached_stop_words.update(("and","I","A","http","And","So","arnt","This","When","It","many",
        "Many","so","cant","Yes","yes","No","no","These","these","mailto","regards","ayanna","like","email",))

    def remove_stop_words(self, str):
        ...

    def punct(self, text):
        ...

    def clean_html(self, text):
        ...

    def remove_links(self, link):
        ...

    def remove_special_characters(self, text):
        ...

    def remove_(self, link):
        ...

    def remove_digits(self, text):
        ...

    def lower(self, text):
        ...

    def email_address(self, text):
        ...

    def non_ascii(self, s):
        ...

    def __iter__(self):
        ...

```

### 3. Huấn luyện mô hình:
### **Sử dụng lại pre-trained tokenizer và DistilBert model:**

Mô hình DistilBERT: sử dụng kỹ thuật chắt lọc (distillation) bằng cách sử dụng thuật toán xấp xỉ trong
thống kê Bayes là Kulback Leiber để xấp xỉ các kiến trúc mô hình mạng nơ-ron lớn bằng các các mạng có
kiến trúc nhỏ hơn. DistilBERT có kiến trúc giảm đi so với BERT 40%. Mô hình DistilBERT được đề xuất
để tăng tốc độ tính toán mà vẫn giữ được độ chính xác và hiệu quả của mô hình

Chúng em sử dụng từ model có sẵn “manishiitg/distilbert-resume-parts-classify” ([https://huggingface.co/manishiitg/distilbert-resume-parts-classify](https://huggingface.co/manishiitg/distilbert-resume-parts-classify)) và tinh chỉnh lại từ model trên.

```jsx
tokenizer = AutoTokenizer.from_pretrained("manishiitg/distilbert-resume-parts-classify")
bert_model = TFDistilBertForSequenceClassification.from_pretrained("manishiitg/distilbert-resume-parts-classify",from_pt=True)
```

### Chia tập dữ liệu đã được xử lý trước thành tập huấn luyện và tập kiểm tra:

```jsx
 df_train, df_test = train_test_split(
        df.data, test_size=0.3, shuffle=True, random_state=101
    )

```

### Text tokenization:

```jsx
     x_train = self.tokenizer(
            text=self.df_train.x.tolist(),
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="tf",
            return_attention_mask=True,
            return_token_type_ids=False,
            verbose=1,
        )
        x_test = self.tokenizer(
            text=self.df_test.x.tolist(),
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="tf",
            return_attention_mask=True,
            return_token_type_ids=False,
            verbose=1,
        )
```

### Xác định kiến trúc mô hình:

Mô hình này tổng cộng được gắn thêm 13 lớp, bao gồm các lớp biến đổi embedding từ BERT, các lớp Dense để học đặc trưng, các lớp chuẩn hóa và Dropout để điều chỉnh và ngăn chặn overfitting, và lớp đầu ra để dự đoán phân loại vào 25 nhãn khác nhau là các  role  trong dataset.  Mỗi lớp được thiết kế để học một cách hợp lý các đặc trưng từ dữ liệu và chuẩn bị dữ liệu để đưa ra dự đoán chính xác.

```jsx
input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name="input_ids")
attention_masks = Input(shape=(self.max_length,), dtype=tf.int32, name="attention_mask")

embeddings = self.bert_model(input_ids, attention_mask=attention_masks)[0]

output = Flatten()(embeddings)
output = Dense(units=1024, activation="relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.25)(output)
output = Dense(units=512, activation="relu")(output)
output = Dropout(0.25)(output)
output = Dense(units=256, activation="relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.25)(output)
output = Dense(units=128, activation="relu")(output)
output = Dropout(0.25)(output)
output = Dense(units=64, activation="relu")(output)
output = Dense(units=25, activation="softmax")(output)

self.bert_model = Model(inputs=[input_ids, attention_masks], outputs=output)
self.bert_model.layers[2].trainable = True
```
Plot : 

![image](https://github.com/SnowyField1906/resume-classification-bert/assets/57946382/108a715e-70f5-431d-9f70-405a58337a6a)

### Biên dịch mô hình:

```jsx
 model.compile(
        optimizer=Adam(learning_rate=5e-5, epsilon=2e-8, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=SparseCategoricalAccuracy("balanced_accuracy"),
    )
```

### Huấn luyện mô hình Distil-Bert đã được tinh chỉnh:

```jsx
loss, acc = model.train(
        [
            EarlyStopping(
                monitor="val_balanced_accuracy",
                patience=250,
                verbose=1,
                mode="max",
                restore_best_weights=True,
            ),
        ]
    )
```
### 3. Đánh giá hiểu xuất mô hình:
### Minh họa hiệu xuất mô hình:

```jsx
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(t.history['loss'],'r',label='train loss')
ax1.plot(t.history['val_loss'],'b',label='test loss')
ax1.set_xlabel('No. of Epochs')
ax1.set_ylabel('Categorical Crossentropy Loss')
ax1.set_title('Loss Graph')
ax1.legend();
ax1.grid(True)
ax2.plot(t.history['balanced_accuracy'],'r',label='train accuracy')
ax2.plot(t.history['val_balanced_accuracy'],'b',label='test accuracy')
ax2.set_xlabel('No. of Epochs')
ax2.set_ylabel('Balanced Categorical Accuracy')
ax2.set_title('Accuracy Graph')
ax2.legend();
ax2.grid(True)
plt.tight_layout()
plt.savefig("./model/assets/loss_accuracy_graph.png")
```

![image](https://github.com/SnowyField1906/resume-classification-bert/assets/57946382/b376615c-84eb-499e-bb00-bdc6275bfa85)


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



### Confusion Matrix & Classification Report:
```jsx
test_predictions
print("Confusion Matrix:")
print(confusion_matrix(test_df.Category,test_predictions))
print("Classification Report:")
print(classification_report(test_df.Category,test_predictions))
```

Confusion Matrix: 
![image](https://github.com/SnowyField1906/resume-classification-bert/assets/57946382/b6f76870-b947-4090-8d14-3c1e710b36a7)



## III - Xây dựng ứng dụng dựa trên mô hình đã huấn luyện:
<a name="xaydungungdung"></a>

### 1 - Cấu trúc folder:
<a name="cautrucfolder"></a>
```jsx
- resume-classification-bert/
  - doc/
    - doc.md
  - model/
    - assets/  
    - data_frame.py
    - distri_bert_classify.py
    - text_preprocessor.py
  ...
  - app.py
  - main.py
  - README.md
  ```
__doc.md__ : Báo cáo đồ án

__assets/__ : Thư mục chứa dữ liệu training và data visualization

__model/__ : Thư mục src code của mô hình 

__data_frame.py__ : file code chuẩn bị, xử lí dữ liệu, chứa các phương thức để tính toán thêm thông tin cho dữ liệu, tiền xử lý văn bản và chuyển đổi nhãn, cùng với việc xây dựng một giao diện để truy xuất thông tin về nhãn sau khi huấn luyện
    
__distri_bert_classify.py__: file code của mô hình distribert
    
__text_preprocessor.py__: file code tiền xử lí dữ liệu

__app.py__: Thông tin API

__main.py__: Chương trình chính
    
### 2 - Setup môi trường và chạy chương trình:
<a name="setupandrun"></a>

### Yêu cầu:

- Cài đặt sẵn môi trường `python` và `pip` 
- Cài đặt `miniconda` để quản lí môi trường ( recommended )

### Setup môi trường

```bash
# Tạo môi trường conda mới
conda create -n resume-classification-bert python=3.10.13
conda activate resume-classification-bert

# Cài đặt dependencies
pip install -r requirements.txt
```

### Huấn luyện mô hình
#### Cách 1: Huấn luyện trực tiếp

```bash
python main.py
```

#### Cách 2: Download mô hình đã được huấn luyện sẵn:
Do quá trình huấn luyện có thể tốn nhiều giờ nên, chúng em đã chuẩn bị sẵn dữ liệu đã được train, thầy cô có thể tải xuống và sử dụng ngay ( Lưu ý đặt file đã tải xuống ở /model/assets )

[drive.google.com/file/d/1r93iNQdgTqOlgDOm37oNESkX9PoXr_Ln](https://drive.google.com/file/d/1r93iNQdgTqOlgDOm37oNESkX9PoXr_Ln) 

## Chạy chương trình:
### Chạy frontend:


### Chạy backend server

```bash
python app.py
```

- Default port: 5000
- API endpoint:
  - `POST /train` - `{loss: string, acc: string}`: Train the model.
  - `GET /process?content={string}` - `{[role: string]: string}`: Read content from PDF file and return the classification result.

### Alternative approach: Chạy trực tiếp không qua backend

```bash
python main.py "path/to/pdf/file"
```

Example:

```bash
python main.py "./model/assets/resume.pdf"
```

```json
{
    "Data Science": 0.01,
    "HR": 0.0,
    "Advocate": 0.0,
    "Arts": 0.0,
    "Web Designing": 0.0,
    "Mechanical Engineer": 0.0,
    "Sales": 0.0,
    "Health and fitness": 0.0,
    "Civil Engineer": 0.0,
    "Java Developer": 0.0,
    "Business Analyst": 0.0,
    "SAP Developer": 0.0,
    "Automation Testing": 0.0,
    "Electrical Engineering": 0.0,
    "Operations Manager": 0.0,
    "Python Developer": 0.0,
    "DevOps Engineer": 0.04,
    "Network Security Engineer": 0.0,
    "PMO": 0.0,
    "Database": 0.0,
    "Hadoop": 0.0,
    "ETL Developer": 0.0,
    "DotNet Developer": 0.0,
    "Blockchain": 99.96,
    "Testing": 0.0
}
```

## IV - Tham khảo: 
<a name="thamkhao"></a>
1. [github.com/huggingface/transformers](https://github.com/huggingface/transformers)

  


