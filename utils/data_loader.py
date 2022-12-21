
# import json
# for current_dataset in ['ArMIS','MD-Agreement','ConvAbuse', 'HS-Brexit'] :                        # loop on datasets
#   for current_split in ['train','dev']:                                                           # loop on splits
#     current_file = './'+current_dataset+'_dataset/'+current_dataset+'_'+current_split+'.json' 
#     data = json.load(open(current_file,'r', encoding = 'UTF-8'))                                   
    



# #===== snippet 2: how to read data and save text, soft evaluation and hard evaluation in a different file for each dataset/split
# # with these few lines you can loop across all datasets and splits (here only the train) and
# # extract (and print) the info you need 
# # here we print: dataset,split,id,lang,hard_label,soft_label_0,soft_label_1,text in a tab separated format
# # note: each item_id in the dataset for each split is numbered starting from "1"

# import json

# print("Dataset\tSplit\tId\tLang\tHard_label\tSoft_label_0\tSoft_label_1\tText")                   # print header

# for current_dataset in ['ArMIS','MD-Agreement','ConvAbuse', 'HS-Brexit']:                         # loop on datasets
#   for current_split in ['train']:                                                                 # loop on splits, here only train 
#     current_file = './'+current_dataset+'_dataset/'+current_dataset+'_'+current_split+'.json'     # current file 
#     data = json.load(open(current_file,'r', encoding = 'UTF-8'))                                  # load data 
#     for item_id in data:                                                                          # loop across items for the loaded datasets                                                                                                  
#       text = data[item_id]['text']        
#       text = text.replace('\t',' ').replace('\n',' ').replace('\r',' ')                           # remove tabs and similar from text, so we can have everything on a line 
#       print('\t'.join([current_dataset, current_split, item_id, data[item_id]['lang'], str(data[item_id]['hard_label']), str(data[item_id]['soft_label']["0"]), str(data[item_id]['soft_label']["1"]), text]))



# #===== snippet 3: ConvAbuse dataset text from string to conversation
# # only for the ConvAbuse dataset, the field "text" is a conversation that for representational purposes has been "stringified". To put it back to it's conversation form you can run

# # current_dataset = 'ConvAbuse'
# # for current_split in ['train', 'dev']:                                                            # loop on splits, here only train 
# #     current_file = './'+current_dataset+'_dataset/'+current_dataset+'_'+current_split+'.json'     # current file     
# #     f_out = open('./'+ d+ '_'+ k +'_conversation.json','w')
# #     data = json.load(open(current_file,'r', encoding = 'UTF-8'))                                  # load data 
# #     for item_id in data:                                                                          # loop across items for the loaded datasets                                                                                                  
# #         data[item_id]['text'] = json.dumps(data[item_id]['text'])
# #     f_out.write(json.dumps(data, indent = 4))
# #     f_out.close()

import json
current_dataset = "ConvAbuse_train.json"
data = json.load(open(current_dataset,'r', encoding = 'UTF-8'))                                  # load data 
for item_id in data:                                                                          # loop across items for the loaded datasets                                                                                                  
    text = data[item_id]['text']        
    text = text.replace('\t',' ').replace('\n',' ').replace('\r',' ')                           # remove tabs and similar from text, so we can have everything on a line 
    print('\t'.join([current_dataset, item_id, data[item_id]['lang'], str(data[item_id]['hard_label']), str(data[item_id]['soft_label']["0"]), str(data[item_id]['soft_label']["1"]), text]))

