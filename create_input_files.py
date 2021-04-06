from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='face2text',
                       karpathy_json_path='./caption data/dataset_face2text.json',
                       image_folder='./caption data/dataset-face2text/',
                       captions_per_image=2,
                       min_word_freq=2,
                       output_folder='./caption data/',
                       max_len=50)
