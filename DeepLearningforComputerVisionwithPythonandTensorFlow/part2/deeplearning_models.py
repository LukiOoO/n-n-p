#code from https://www.youtube.com/watch?v=cPmjQ9V6Hbk

import tensorflow
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# tensorflow.keras.Sequential
seq_model = tensorflow.keras.Sequential(
    [
        Input(shape=(28,28,1)),
        Conv2D(32, (3,3), activation='relu'),
        Conv2D(64, (3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128, (3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ]
)

# tensorflow.keras.Model : inhert from this class
# tensorflow.keras.Model : inherit from this class
class MyCustomModel(tensorflow.keras.Model):

    def __init__(self):
        super().__init__()

        self.conv1 = Conv2D(32, (3,3), activation='relu')
        self.conv2 = Conv2D(64, (3,3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3,3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()

        self.globalavgpool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')  
     

    def call(self, my_input):

        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x) 

        return x    



# functional approach : function that returns model 
def functional_model():

    my_input = Input(shape=(28,28,1))
    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = tensorflow.keras.Model(inputs=my_input, outputs=x)

    return model


def streesigns_model(nbr_classes):

    my_input = Input(shape=(60,60,3))

    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    #x = Flatten()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(nbr_classes, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

def create_generators(batch_size, train_data_path, val_data_path, test_data_path):

    train_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
        rotation_range=10,
        width_shift_range=0.1
    )

    test_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
    )

    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = test_preprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )

    test_generator = test_preprocessor.flow_from_directory(
        test_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )

    return train_generator, val_generator, test_generator

if __name__ == '__main__':
    
    model = streesigns_model(10)
    model.summary()
