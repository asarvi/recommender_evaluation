
from model.data_utils import CoNLLDataset
from model.aspect_model import ASPECTModel
from model.config import Config
from ABSA.example import ABSA
import tensorflow as tf
import json
import random

def align_data(data):

    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned



def interactive_shell(model , sentence):



        words_raw = sentence.strip().split(" ")

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})


        aspects = aspectsToarray(words_raw ,preds )
        return aspects

# change B-A type to strings...
def aspectsToarray(sentence,preds):
    aspects = []
    for i in range(0 ,len(preds)-2):

        if(preds[i] == 'B-A' and preds[i+1] =='I-A' and preds[i+2] != 'I-A'):


            aspects.append(sentence[i] +" "+sentence[i+1])
        elif(preds[i] == 'B-A' and preds[i+1] =='I-A' and preds[i+2] == 'I-A'):

            aspects.append(sentence[i])
        elif( preds[i] =='B-A' and preds[i+1] =='O'):

            aspects.append(sentence[i])
        elif (preds[i] == 'B-A' and preds[i + 1] == 'B-A'):

            aspects.append(sentence[i] )
    if(preds[len(preds)-1] =='B-A'):
        aspects.append(sentence[len(preds)-1])
    if (preds[len(preds) - 2] == 'B-A'):
        aspects.append(sentence[len(preds) - 2])

    return  aspects

def aspectSentiment(comment ,aspec):
    # from ABSA project imported
    sent = ABSA(comment,aspec)
    return sent


def aspectExtractor(sentence):
    # create instance of config
    config = Config()
    # build model
    model = ASPECTModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)

    # evaluate and interact
    model.evaluate(test)
    preds=interactive_shell(model , sentence)
    return preds



# function to search in dictionary
def search(values, searchFor):
    for k in values:
        if searchFor in k:
            return k
    return None

def countX(lst, x):
    count = 0

    for ele in lst:
        if (ele == x):
            count = count + 1
    return count

#function for search in list
def listContains(list , object):
    j =0
    for i in list:
        if object==i:
            j=1
    if j>0:
        return 1
    elif j==0:
        return 0

def recommend(productname, negativeAspect):
    productFile = open('testData/RecommendedObjects' + productname + '.txt')
    # dictionary to keep the repeatation of products in files
    dic = {}

    listOfProducts = []

    #final recommended objects:

    recomList= []

    for p in productFile:
        temp = p.split('	')
        tempProduct = temp[0]
        listOfProducts.append(tempProduct)

    #print(listOfProducts)

    # count each product
    for l in listOfProducts:
        count =countX(listOfProducts , l)
        dic[l] = count

    # now look for each item
    counter = 0

    productFile = open('testData/RecommendedObjects' + productname + '.txt')
    for n in (dic):

        for m,line in enumerate(productFile):
            if counter <= m < counter + dic.get(n):

                comment = ""
                counter = counter+1
                temp = line.split('	')
                productID = temp[0]
                # split comments from product id
                for j in range(1, len(temp)):
                    comment = comment + temp[j] + " "

                if negativeAspect in comment:
                    comment =comment.replace(negativeAspect,"$t$")
                    #if  aspect was +
                    if aspectSentiment(comment ,negativeAspect)>-1:
                        recomList.append(productID)

                        recomList = list(dict.fromkeys(recomList))

                    elif aspectSentiment(comment, negativeAspect) == -1:
                        if listContains(recomList, productID):
                          recomList.remove(productID)
                          recomList = list(dict.fromkeys(recomList))


                elif negativeAspect not in comment :
                    recomList.append(productID)
                    recomList = list(dict.fromkeys(recomList))


    #hazfe tekraria az list

    print(recomList)
    return recomList


#open testfile
result_file = open('final_result.txt', 'w')
testFile = open('testData/TestFile.txt', 'r')
#array definition
reviewPart =[]
# recall

#read line by line
for line in testFile:
    reviewPart= line.split('%%&%%')

    trueRecommendation = 0
    falseRecommendarion = 0

   # print(reviewPart[1])
    review = reviewPart[1]

    #temp is the array which splits review
    temp = review.split()

    #negative aspect of product
    negativeAspect = temp[0]

    recommends = reviewPart[0].split('    ')[1]
    recommendObjects = recommends.split();
    product = reviewPart[0].split('    ')[0]
    recomlist = recommend(product, negativeAspect)

    for obj in recomlist:

         if listContains(recommendObjects,obj):
             trueRecommendation= trueRecommendation+1
         else:
            falseRecommendarion= falseRecommendarion+1
    print("recall for product " + product + " is")
    print(trueRecommendation/ (trueRecommendation +falseRecommendarion))
    result_file.write("recall for product " + product + " is" + '\n')
    result_file.write(str(trueRecommendation/ (trueRecommendation +falseRecommendarion)) + '\n')
    result_file.write(str(recommendObjects) + '\n')

