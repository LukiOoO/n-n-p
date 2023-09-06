from my_utils import split_data, order_test_set
from deeplearning_models import streesigns_model, create_generators
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

if __name__ == '__main__':

    if False:
        path_to_data = '/mnt/x/programowanie/Pythonex/n-n-p/DeepLearningforComputerVisionwithPythonandTensorFlow/part2/2:24_end/dataset/Train'
        path_to_save_train = '/mnt/x/programowanie/Pythonex/n-n-p/DeepLearningforComputerVisionwithPythonandTensorFlow/part2/2:24_end/dataset/traning_data/train'
        path_to_save_val = '/mnt/x/programowanie/Pythonex/n-n-p/DeepLearningforComputerVisionwithPythonandTensorFlow/part2/2:24_end/dataset/traning_data/val'
        split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)

        path_to_images = '/mnt/x/programowanie/Pythonex/n-n-p/DeepLearningforComputerVisionwithPythonandTensorFlow/part2/dataset/Test'
        path_to_csv = '/mnt/x/programowanie/Pythonex/n-n-p/DeepLearningforComputerVisionwithPythonandTensorFlow/part2/dataset/Test.csv'
        order_test_set(path_to_images, path_to_csv)

    path_to_train = '/mnt/x/programowanie/Pythonex/n-n-p/DeepLearningforComputerVisionwithPythonandTensorFlow/part2/dataset/traning_data/train'
    path_to_val = '/mnt/x/programowanie/Pythonex/n-n-p/DeepLearningforComputerVisionwithPythonandTensorFlow/part2/dataset/traning_data/val'
    path_to_test = '/mnt/x/programowanie/Pythonex/n-n-p/DeepLearningforComputerVisionwithPythonandTensorFlow/part2/dataset/Test'
    batch_size = 64
    epochs = 15
    lr=0.0001

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes

    TRAIN=False
    TEST=True

    if TRAIN:
        path_to_save_model = './Models'
        ckpt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        early_stop = EarlyStopping(monitor="val_accuracy", patience=10)

        model = streesigns_model(nbr_classes)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_generator,
                    epochs=15,
                    batch_size=batch_size,
                    validation_data=val_generator,
                    callbacks=[ckpt_saver, early_stop]
                    )
        
    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()
        
        print('Evaluating validation set')
        model.evaluate(val_generator)
        
        print('Evaluation test set')
        model.evaluate(test_generator)


