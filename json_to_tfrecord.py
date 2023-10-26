import json
import pandas as pd

FILENAME = 'test_names'

# load JSON, convert to CSV, then convert to TFRecord

def json_to_csv(filename):
    f = open(filename)
    data = json.load(f)

    csv_list = []

    '''print(data[0])
    print(data[0]['ann']['bboxes'][0][0])
    print(data[0]['ann']['labels'][0])'''

    for i in data:
        print(i['filename'])
        filename = i['filename']
        width = i['width']
        height = i['height']
        bbox = i['ann']['bboxes'][0]
        label = i['ann']['labels'][0]

        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]

        value = (filename, width, height,
                    label, xmin, ymin, xmax, ymax)

        csv_list.append(value)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    csv_df = pd.DataFrame(csv_list, columns=column_name)

    return csv_df

json_file = FILENAME + '.json'
print(json_file)
    
csv_file = json_to_csv(json_file)
    
csv_file.to_csv(FILENAME + '.csv')