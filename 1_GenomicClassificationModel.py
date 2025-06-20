import torch 
#import vcf
import time
import numpy as np
import pandas as pd
import math
import os
import openpyxl
import time
import matplotlib.pyplot as plt
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import tensorflow as tf
from random import sample
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from keras import optimizers # 2024-04-24
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, LSTM, GRU, Bidirectional, MaxPooling1D, Conv1D, BatchNormalization # 2024-04-24
from sklearn.metrics import accuracy_score
from keras.applications import ResNet50, DenseNet201, VGG19
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, \
    ZeroPadding2D, Add, Dropout, Flatten, concatenate
from keras import Input, Model, layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import warnings

warnings.filterwarnings('ignore')

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# cnn model 1
def Func_model(tmp):
    model = Sequential()
    model.add(Conv2D(filters=16, input_shape=(tmp[0], tmp[1], tmp[2]), kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))

    return model

# cnn model 2
def Func_model_2(tmp):
    model = Sequential()
    model.add(Conv2D(filters=32, input_shape=(tmp[0], tmp[1], tmp[2]), kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    # model.add(Dense(10, activation='softmax'))
    model.add(Dense(5, activation='softmax'))

    return model

# lstm model
def Func_model_RNN_LSTM(tmp):
    model = Sequential()
    model.add(LSTM(32))
    model.add(Dense(5, activation='softmax'))

    return model

#bilstm model
def Func_model_RNN_BiLSTM(tmp):
    model = Sequential()
    model.add(Bidirectional(32))
    model.add(Dense(5, activation='softmax'))

    return model

#gru model
def Func_model_RNN_GRU(tmp):
    model = Sequential()
    model.add(layers.Embedding(input_dim=1000, output_dim=64))
    model.add(GRU(128))
    model.add(Dense(5, activation='softmax'))

    return model

# rnn + cnn model
def Func_model_RNN_CNN(tmp):
    model = Sequential()
    # model.add(Flatten())
    model.add(Conv2D(3, 3, input_shape=(tmp[0], tmp[1], tmp[2]), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(3))
    model.add(Flatten())
    model.add(layers.Embedding(input_dim=16, output_dim=1))
    model.add(LSTM(32))
    model.add(Dense(5, activation='softmax'))
    # model.add(Conv1D(filters=16, input_shape=(tmp[0], tmp[1], tmp[2]), kernel_size=2, padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=2))
    # model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=2))
    # model.add(Flatten())
    # model.add(layers.Embedding(input_dim=16, output_dim=4))
    # model.add(LSTM(128))
    # model.add(Dense(4, activation='softmax'))
    # model.summary()
    return model

#optimizer
def Func_model_opt(model, lr_):
    nadam = optimizers.Nadam(lr=lr_)
    model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# resnet50 model
def model_resnet50(tmp):
    model_ = ResNet50(input_tensor=Input(shape=(tmp[0], tmp[1], tmp[2])), include_top=False, weights=None,
                      pooling='max')
    x = model_.output
    x = Dense(5, activation='softmax')(x)
    model_ = Model(model_.input, x)

    return model_

# vgg19 model
def model_vgg19(tmp):
    model_ = VGG19(input_tensor=Input(shape=(tmp[0], tmp[1], tmp[2])), include_top=False, weights=None, pooling='max')
    x = model_.output
    x = Dense(5, activation='softmax')(x)
    model_ = Model(model_.input, x)

    return model_

# alexnet model
def model_alexnet(tmp, tmp_categ):
    model = Sequential()
    model.add(Conv2D(filters=96, input_shape=(tmp[1], tmp[2], tmp[3]), kernel_size=(11, 11), strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Flatten())
    model.add(Dense(4096, input_shape=(256 * 256 * 3,)))
    model.add(Activation('relu'))

    model.add(Dropout(0.4))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(tmp_categ))
    model.add(Activation('softmax'))

    return model

# inception module
def inception_module(x, o_1=64, r_3=64, o_3=128, r_5=16, o_5=32, pool=32):
    x_1 = Conv2D(o_1, 1, padding='same')(x)
    x_2 = Conv2D(r_3, 1, padding='same')(x)
    x_2 = Conv2D(o_3, 3, padding='same')(x_2)

    x_3 = Conv2D(r_5, 1, padding='same')(x)
    x_3 = Conv2D(o_5, 5, padding='same')(x_3)
    x_4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    x_4 = Conv2D(pool, 1, padding='same')(x_4)

    return concatenate([x_1, x_2, x_3, x_4])

# googlenet model
def model_googlenet(tmp, tmp_categ):
    input_ = Input(shape=(tmp[1], tmp[2], tmp[3]))
    x = Conv2D(64, 7, strides=2, padding='same')(input_)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 1, strides=1)(x)
    x = Conv2D(192, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = inception_module(x, o_1=64, r_3=64, o_3=128, r_5=16, o_5=32, pool=32)
    x = inception_module(x, o_1=128, r_3=128, o_3=192, r_5=32, o_5=96, pool=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = inception_module(x, o_1=192, r_3=96, o_3=208, r_5=16, o_5=48, pool=64)
    x = inception_module(x, o_1=160, r_3=112, o_3=224, r_5=24, o_5=64, pool=64)
    x = inception_module(x, o_1=128, r_3=128, o_3=256, r_5=24, o_5=64, pool=64)
    x = inception_module(x, o_1=112, r_3=144, o_3=288, r_5=32, o_5=64, pool=64)
    x = inception_module(x, o_1=256, r_3=160, o_3=320, r_5=32, o_5=128, pool=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = inception_module(x, o_1=256, r_3=160, o_3=320, r_5=32, o_5=128, pool=128)
    x = inception_module(x, o_1=384, r_3=192, o_3=384, r_5=48, o_5=128, pool=128)
    x = layers.AveragePooling2D(pool_size=(7, 7), strides=1)(x)
    x = layers.Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(units=tmp_categ, activation='softmax')(x)
    model_ = Model(input_, x)

    return model_


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

# 결과 파일 폴더 생성
Today = datetime.today().strftime("%Y%m%d")
dir = '/mnt/d/Documents'
if not (os.path.isdir(os.path.join(dir, 'All_validation_result_'+ Today))):
    os.makedirs(os.path.join(dir, 'All_validation_result_' + Today))
save_dir = os.path.join(dir, 'All_validation_result_' + Today)
print(': Success to make folder for result file')


# 결과 요약(csv 파일)
save_dir_csv = os.path.join(save_dir, 'All_process_result'+ Today + '.csv')

# EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=5)
acc = []

#columns = ['BIG', 'CER', 'F1', 'PIM','WILD']
columns = ['Single', 'SemiDouble', 'Double']
sample_result = pd.DataFrame(columns=columns)

for idx_scenario in range(0, 14):
    # 0 -> 전체이미지(Ref)
    # 1 -> Chr 별
    # 2 - 11 -> Chr의 Piece 별

    result_prob = []
    max_acc = 0.80

    # Chromosome 별로 CNN 수행
    if idx_scenario == 0:
        dataset_dir = '/BiO/Majung_work_2022/10_Hibiscus_GeneticMap/10_Data/Dataset_ALL_20240424'
    elif idx_scenario == 1:
        dataset_dir = '/BiO/Majung_work_2022/10_Hibiscus_GeneticMap/10_Data/Dataset_CHR_20240424'
    else:
        max_acc = 0.50
        if idx_scenario > 10:
            tmp_chr = 'A' + str(idx_scenario - 1)
        else:
            tmp_chr = 'A0' + str(idx_scenario - 1)
        dataset_dir = os.path.join('/BiO/Majung_work_2022/10_Hibiscus_GeneticMap/10_Data/Dataset_CHR_CROP_20240423', tmp_chr)


    # unique 개수 확인을 위해 image 폴더 불러오기
    data_generator = ImageDataGenerator(rescale=1./255)
    image_all = data_generator.flow_from_directory(
        directory=os.path.join(dataset_dir),
        batch_size=1,
        color_mode='rgb',
        class_mode='categorical')
    if idx_scenario == 0:
        chr = ['Reference']
    else:
        chr = list(image_all.class_indices.keys())

    t = 1

    for idx_chr in range(len(chr)):

        if idx_scenario == 0:
            data_generator = ImageDataGenerator(rescale=1. / 255)
            image_all = data_generator.flow_from_directory(
                    directory=os.path.join(dataset_dir),
                    batch_size=1,
                    target_size=(200, 200),
                    color_mode='rgb',
                    class_mode='categorical')
        else:
            data_generator = ImageDataGenerator(rescale=1. / 255)
            image_all = data_generator.flow_from_directory(
                    directory=os.path.join(dataset_dir, chr[idx_chr]),
                    batch_size=1,
                    target_size=(200, 200),
                    color_mode='rgb',
                    class_mode='categorical')

        # Dataset 분할을 위한 파라미터 셋팅
        Ratio_test_set = 0.2
        Ratio_val_set = 0.2

        while True:
            # Split 8:2(Train/Test)
            # train/validation/test set split
            random_number_val = []
            random_number_test = []
            tmp = 0
            for idx_categ in range(len(image_all.class_indices)):
                random_num_max = sum(image_all.classes == idx_categ)
                random_num_max_val = math.ceil(sum(image_all.classes == idx_categ) * (1 - Ratio_test_set))
                tmp_random_number_test = list(range(random_num_max_val, random_num_max))
                tmp_random_number_val = sample(range(0, random_num_max_val), math.ceil(random_num_max_val * Ratio_val_set))
                random_number_val.extend(tmp_random_number_val + np.ones(shape=(len(tmp_random_number_val))) * tmp)
                random_number_test.extend(tmp_random_number_test + np.ones(shape=(len(tmp_random_number_test))) * tmp)

                tmp = tmp + random_num_max

            # train/val/test set을 위한 변수
            x_train = []
            y_train = []
            x_val = []
            y_val = []
            x_test = []
            y_test = []

            # 파일명 저장을 위한 변수
            train_set = []
            val_set = []
            test_set = []
            tmp = 0

            for idx_all_categ in range(len(image_all)):
                tmp_value, tmp_categ = image_all.next()
                # test set
                if tmp in random_number_val:
                    x_val.append(tmp_value)
                    y_val.append(tmp_categ)
                    val_set.append(image_all.filenames[tmp].split('/')[1].split('.')[0])
                elif tmp in random_number_test:
                    x_test.append(tmp_value)
                    y_test.append(tmp_categ)
                    test_set.append(image_all.filenames[tmp].split('/')[1].split('.')[0])
                # training set
                else:
                    x_train.append(tmp_value)
                    y_train.append(tmp_categ)
                    train_set.append(image_all.filenames[tmp].split('/')[1].split('.')[0])
                tmp += 1

            x_train = np.array(x_train).reshape(np.array(x_train).shape[0], np.array(x_train).shape[2], np.array(x_train).shape[3], np.array(x_train).shape[4])
            y_train = np.array(y_train).reshape(np.array(y_train).shape[0], np.array(y_train).shape[2])
            x_val = np.array(x_val).reshape(np.array(x_val).shape[0], np.array(x_val).shape[2], np.array(x_val).shape[3], np.array(x_val).shape[4])
            y_val = np.array(y_val).reshape(np.array(y_val).shape[0], np.array(y_val).shape[2])
            x_test = np.array(x_test).reshape(np.array(x_test).shape[0], np.array(x_test).shape[2], np.array(x_test).shape[3], np.array(x_test).shape[4])
            y_test = np.array(y_test).reshape(np.array(y_test).shape[0], np.array(y_test).shape[2])
            print(len(x_train), len(x_val), len(x_test))

            # CNN 모델 및 옵티마이저
            #with tf.device('/GPU'):
            with tf.device('/CPU'):
                # print(train_image.image_shape)
                save_model = save_dir + '/best_model_' + str(idx_scenario) + '_' + str(idx_chr) + '.h5'
                mc = ModelCheckpoint(save_model, monitor='val_loss', mode='min', save_best_only=True)
                model = Func_model_2(image_all.image_shape)
                model = Func_model_opt(model, 0.0001)
                history = model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=0, validation_data=(x_val, y_val), callbacks=[es, mc])
                # model.summary()

                # prediction 결과
                val_loss, val_acc = model.evaluate(x_val, y_val)
                test_loss, test_acc = model.evaluate(x_test, y_test)
                print(val_acc, test_acc)

            # 결과 파일 저장
            if test_acc > max_acc:
                result_pred = list(model.predict(x_test))
                #result_classes = list(model.predict_classes(x_test))
                pd_tmp_result = pd.DataFrame(result_pred)
                pd_test_set = pd.DataFrame(test_set)
                # pd_class = pd.DataFrame(result_classes)
                pd_final_result = pd.concat([pd_test_set, pd_tmp_result], axis=1)
                idx = list(image_all.class_indices)
                idx.insert(0, 'Sample')
                # idx.append('Predict class')
                # pd_final_result.columns = idx
                # pd_final_result.to_csv(save_dir_csv, header=True, index=False)

                cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1))
                ['Single', 'SemiDouble', 'Double']
                #disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=['BIG', 'CER', 'F1', 'PIM','WILD'])
                disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=['Single', 'SemiDouble', 'Double'])
                disp = disp.plot(cmap=plt.cm.Blues, values_format='g', xticks_rotation='vertical')
                plt.xticks(rotation=45)
                plt.savefig(save_dir + '/Confusion_matrix_'+ str(idx_scenario) + '_' + str(idx_chr) +'.jpg', dpi=400, bbox_inches='tight')

                tmp_sample_result = model.predict(x_test)[0]
                A = pd.DataFrame(np.reshape(tmp_sample_result, (1, len(tmp_sample_result))), columns = columns)
                sample_result = sample_result.append(A)
                print(sample_result)
                sample_result.to_csv(save_dir_csv, header=True, index=False)

                break

        print(':Success to make model and test of Scenario_' + str(idx_scenario) + 'and Chr_' + str(idx_chr))

############################################################
############################################################
