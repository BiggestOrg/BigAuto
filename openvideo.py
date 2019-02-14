import cv2
import time
from utils.preprocessing import *
from network.lane_segmentator import Segmentator


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_gpu', 1, 'Number of GPUs to use.')
model_dir = './model'
input_size = (1920, 1080)
image_size = (1920, 1080)

sess = tf.Session()
input_image = tf.placeholder(dtype=tf.float32, shape=[None, input_size[1], input_size[0], 3])

segmentation = Segmentator(
    params={
        'base_architecture': 'resnet_v2_50',
        'batch_size': 1,
        'fine_tune_batch_norm': False,
        'num_classes': 2,
        'weight_decay': 0.0001,
        'output_stride': 16,
        'batch_norm_decay': 0.9997
    }
)

logits = segmentation.network(inputs=input_image, is_training=False)

predict_classes = tf.expand_dims(
    tf.argmax(logits, axis=3, output_type=tf.int32),
    axis=3
)

variables_to_restore = tf.contrib.slim.get_variables_to_restore()
get_ckpt = tf.train.init_from_checkpoint(
    ckpt_dir_or_file='./model',
    assignment_map={v.name.split(':')[0]: v for v in variables_to_restore}
)

sess.run(tf.global_variables_initializer())

v = cv2.VideoCapture("./data/test10.mp4")
print(v.get(cv2.CAP_PROP_FRAME_WIDTH))
print(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = v.get(cv2.CAP_PROP_FPS)
i = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('./output/output10.avi', fourcc, fps, image_size)

while(True):
    ret, frame = v.read()

    if not ret:
        break

    i+=1

    # img = frame[300:840, 0:1920]

    cv2.imshow("source", frame)

    predictions = sess.run(
        predict_classes,
        feed_dict={
            input_image: np.expand_dims(frame, 0)
        }
    )
    img_test = predictions[0]
    img_test = (img_test * 255).astype(np.uint8)
    img_test = np.squeeze(img_test)
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    print(t, i, img_test.shape)
    # img_test = np.pad(img_test, ((300,240),(0,0)), 'constant', constant_values=(0, 0))
    img_test = cv2.cvtColor(img_test, cv2.COLOR_GRAY2BGR)
    cv2.imshow("bin", img_test)
    writer.write(img_test)
    cv2.waitKey(1)
    # cv2.imwrite("./images/" + str(i) + ".jpg", img_test)
    #print(img_test.shape)

    #red_image = np.transpose(np.tile(predictions[0], [1, 1, 3]), [2, 0, 1])
    #red_image = red_image * 255
    #red_image = np.transpose(red_image, [1, 2, 0])
    #overlay = red_image.astype(np.uint8)

    #cv2.imshow("result", overlay)

writer.release()
v.release()
cv2.destroyAllWindows()
