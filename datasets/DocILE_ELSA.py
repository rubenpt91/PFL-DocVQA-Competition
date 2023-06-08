
import os
import utils
from PIL import Image

import numpy as np
from datasets.BaseDataset import BaseDataset


class DocILE_ELSA(BaseDataset):

    def __init__(self, imbd_dir, images_dir, split, kwargs, indexes=None):
        super(DocILE_ELSA, self).__init__(imbd_dir, images_dir, 'oracle', split, kwargs, indexes)  # TODO Remove the Oracle and let one single dataset

    def __getitem__(self, idx):
        record = self.imdb[idx]

        question = record["question"]
        answers = list(set(answer.lower() for answer in record['answers'])) if 'answers' in record else None
        answer_page_idx = record['answer_page_idx']

        """ Prepared for Single Page <====
        page_context = " ".join([word.lower() for word in record['ocr_tokens'][answer_page_idx]])
        context_page_corresp = [answer_page_idx] * len(record['ocr_tokens'][answer_page_idx])

        if self.get_raw_ocr_data:
            if len(record['ocr_tokens'][answer_page_idx]) == 0:
                words = []
                boxes = np.empty([0, 4])
                                 
            else:
                words = [word.lower() for word in record['ocr_tokens'][answer_page_idx]]
                boxes = np.array([bbox for bbox in record['ocr_normalized_boxes'][answer_page_idx]])

        if self.use_images:
            image_names = os.path.join(self.images_dir, "{:s}".format(record['image_name'][answer_page_idx]))
            images = Image.open(image_names).convert("RGB")
        """

        num_pages = record['total_doc_pages']
        if True:  # self.page_retrieval == 'concat':
            context = ""
            context_page_corresp = []
            for page_ix in range(num_pages):
                page_context = " ".join([word.lower() for word in record['ocr_tokens'][page_ix]])
                context += " " + page_context
                context_page_corresp.extend([-1] + [page_ix]*len(page_context))

            context = context.strip()
            context_page_corresp = context_page_corresp[1:]

            if self.get_raw_ocr_data:
                words, boxes = [], []
                for page_ix in range(num_pages):
                    if len(record['ocr_tokens'][page_ix]) == 0:
                        boxes.append(np.empty([0, 4]))
                        continue

                    words.extend([word.lower() for word in record['ocr_tokens'][page_ix]])
                    boxes.append(np.array(record['ocr_normalized_boxes'][page_ix]))

            if self.use_images:
                image_names = [os.path.join(self.images_dir, "{:s}.jpg".format(image_name)) for image_name in record['image_name']]
                images = [Image.open(img_path).convert("RGB") for img_path in image_names]
                images, boxes = utils.create_grid_image(images, boxes)

        start_idxs, end_idxs = self._get_start_end_idx(context, answers)

        sample_info = {
            # 'question_id': "{:s}_{:d}".format(record['set_name'], idx),
            'question_id': record.get('question_id', "{:s}-{:d}".format(record['set_name'], idx)),
            'questions': question,
            'contexts': context,
            'context_page_corresp': context_page_corresp,
            'answers': answers,
            'answer_page_idx': answer_page_idx,
        }

        if self.use_images:
            sample_info['image_names'] = image_names
            sample_info['images'] = images

        if self.get_raw_ocr_data:
            sample_info['words'] = words
            sample_info['boxes'] = boxes
            sample_info['num_pages'] = num_pages

        else:  # Information for extractive models
            # sample_info['context_page_corresp'] = context_page_corresp
            sample_info['start_indxs'] = start_idxs
            sample_info['end_indxs'] = end_idxs

        if self.get_doc_id:
            sample_info['doc_id'] = [record['image_name'][page_ix] for page_ix in range(first_page, last_page)]

        return sample_info

    def get_pages(self, sample_info):
        # Most of the documents have only 1 page, and maximum 3. Therefore, use always all the pages.
        first_page = sample_info['pages'][0]
        last_page = sample_info['pages'][-1] + 1
        return first_page, last_page



if __name__ == '__main__':
    dude_dataset = DocILE_ELSA("/SSD/Datasets/DocILE/elsa_imdb/", split='val')
