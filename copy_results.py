import json
import pandas as pd
import pyperclip

filename = 'official-matrix-results/unprejudiced_full.json'

with open(filename, encoding='utf-8', mode='r') as f:
    data = json.load(f)

    copy_scores = [
        [
            data['average']['intrasentence']['overall']['LM Score'],
            data['average']['intrasentence']['overall']['SS Score'],
            data['average']['intrasentence']['overall']['ICAT Score'],
            data['average']['intrasentence']['gender']['LM Score'],
            data['average']['intrasentence']['gender']['SS Score'],
            data['average']['intrasentence']['gender']['ICAT Score'],
            data['average']['intrasentence']['profession']['LM Score'],
            data['average']['intrasentence']['profession']['SS Score'],
            data['average']['intrasentence']['profession']['ICAT Score'],
            data['average']['intrasentence']['race']['LM Score'],
            data['average']['intrasentence']['race']['SS Score'],
            data['average']['intrasentence']['race']['ICAT Score'],
            data['average']['intrasentence']['religion']['LM Score'],
            data['average']['intrasentence']['religion']['SS Score'],
            data['average']['intrasentence']['religion']['ICAT Score'],
        ],
        # [
        #     data['average']['intrasentence']['overall']['LM Score'],
        #     data['average']['intrasentence']['overall']['SS Score'],
        #     data['average']['intrasentence']['overall']['ICAT Score'],
        # ],
        # [
        #     data['average']['intrasentence']['gender']['LM Score'],
        #     data['average']['intrasentence']['gender']['SS Score'],
        #     data['average']['intrasentence']['gender']['ICAT Score'],
        # ],
        # [
        #     data['average']['intrasentence']['profession']['LM Score'],
        #     data['average']['intrasentence']['profession']['SS Score'],
        #     data['average']['intrasentence']['profession']['ICAT Score'],
        # ],
        # [
        #     data['average']['intrasentence']['race']['LM Score'],
        #     data['average']['intrasentence']['race']['SS Score'],
        #     data['average']['intrasentence']['race']['ICAT Score'],
        # ],
        # [
        #     data['average']['intrasentence']['religion']['LM Score'],
        #     data['average']['intrasentence']['religion']['SS Score'],
        #     data['average']['intrasentence']['religion']['ICAT Score'],
        # ],
    ]

    copy_string = ''
    for copy_category_scores in copy_scores:
        copy_string += '\t'.join([str(number)
                                 for number in copy_category_scores])
        copy_string += '\n'

    pyperclip.copy(copy_string)
    print('Scores copied for pasting in Google Sheets!')
