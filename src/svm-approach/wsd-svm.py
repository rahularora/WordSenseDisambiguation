import sys
import re
import pdb
import math
import nltk
import xml.dom.minidom
import random
from xml.dom.minidom import Node

TOTAL_COLLOCATION_FEATURE = 11

gTrainingFile = "../../wsd-data/train.data"
gTestingFile = "../../wsd-data/test.data"
gDictionaryFile = "../../dictionary-mapping2.xml"
gTrainOutputFile = "../../training-model.txt"
gOutFileList = {};
gSent_detector = None;
gDictionary = [];
gTrainingList = [];
gTestingList = [];
gSingleWordFeature = {};
gCollocationFeatures = [];
gTraining = None;
fT = None;

gPOSTAGS = {"ADJP":1,"-ADV":2,"ADVP":3,"-BNF":4,"CC":5,"CD":6,"CLF":7,"-CLR":8,"CONJP":9,"-DIR":10,"DT":11,"-DTV":12,"EX":13,"-EXT":14,"FRAG":15,"FW":16,"-HLN":17,"IN":18,"INTJ":19,"JJ":20,"JJR":21,"JJS":22,"-LGS":23,"-LOC":24,"LS":25,"LST":26,"MD":27,"-MNR":28,"NAC":29,"NN":30,"NNS":31,"NNP":32,"NNPS":33,"-NOM":34,"NP":35,"NX":36,"PDT":37,"POS":38,"PP":39,"-PRD":40,"PRN":41,"PRP":42,"-PRP":43,"PRP$":44,"PRT":45,"-PUT":46,"QP":47,"RB":48,"RBR":49,"RBS":50,"RP":51,"RRC":52,"S":53,"SBAR":54,"SBARQ":55,"-SBJ":56,"SINV":57,"SQ":58,"SYM":59,"-TMP":60,"TO":61,"-TPC":62,"-TTL":63,"UCP":64,"UH":65,"VB":66,"VBD":67,"VBG":68,"VBN":69,"VBP":70,"VBZ":71,"-VOC":72,"VP":73,"WDT":74,"WHADJP":75,"WHADVP":76,"WHNP":77,"WHPP":78,"WP":79,"WP$":80,"WRB":81,"X":82,".":83,",":84};

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


class TargetWord:
    m_word = "";
    m_wordBase = "";
    m_wordPos = '';
    m_senses = [];
    m_ptrWord = None;   # reference to ContextWord object
    m_context = None;
    m_singleWordFeatures = {};
    m_collocationFeatures = [];
    m_line = ""; # for segmenting into test and train dataset
    
    def __init__(self):
        self.m_word = "";
        self.m_wordBase = "";
        self.m_wordPos = '';
        self.m_collocationFeatures = [];
        self.m_line = "";
        for i in range(0, 12):
            self.m_collocationFeatures.append({});


class Sense:
    m_senseId = "";
    m_targetWord = None;
    m_synsetsList = [];
    m_glossesList = [];

    def __init__(self):
        self.m_senseId = 0;
        self.m_synsetsList = [];
        self.m_glossesList = [];

#describes the context

class TargetWordContext:
    m_contextWordList = [];
    m_targetWord = None;
    m_contextText = "";

    def __init__(self):
        self.m_contextWordList = [];
        self.m_contextPosList = [];

# describes the word in the context
class ContextWord:
    m_index = -1;
    m_sentIndex = -1;
    m_word = "";
    m_wordPos = "";
    m_stemWord = "";
    m_targetWordContext = None;
    m_bSentEnd = False;

    def __init__(self):
        self.m_word = "";
        self.m_wordPos = "";

class Training:
    m_M2 = 0;
    m_CollocationList = []; # its a 2D array, first index is collocation feature type and second is list of phrases
    m_SingleWordList = {};

    def __init__(self):
        self.m_SingleWordList = {};
        self.m_CollocationList = [];
        for i in range(0, TOTAL_COLLOCATION_FEATURE + 1):
            self.m_CollocationList.append({});

    def OutputSVMFormat(self, binaryAllFeatureValueList, targetWord, bTest):
        global gOutFileList;
        fname = "train";
        if bTest == True:
            fname = "test";
        featureStr = "";
        for i in range(0, len(binaryAllFeatureValueList)):
            if binaryAllFeatureValueList[i] > 0:
            #featureStr = featureStr + " " + str(binaryAllFeatureValueList[i]);
                featureStr = featureStr + " " + str(i+1) + ":" + str(binaryAllFeatureValueList[i]);

        if gOutFileList.has_key(targetWord.m_wordBase) == False:
            gOutFileList[targetWord.m_wordBase] = {};
        if gOutFileList[targetWord.m_wordBase].has_key(targetWord.m_wordPos) == False:
            gOutFileList[targetWord.m_wordBase][targetWord.m_wordPos] = open(targetWord.m_wordBase + '.' + targetWord.m_wordPos + '.' + fname + '.txt', 'w');
        fileN = gOutFileList[targetWord.m_wordBase][targetWord.m_wordPos];

        count = 0;
        if bTest == False:
            for sense in targetWord.m_senses:
                count = count + 1;
                if sense == 1:
                    #featureStrInst = "a" + str(count + 1) + "a" + featureStr + "\n";
                    #featureStrInst = featureStr + " " + str(count) + "\n";
                    featureStrInst = str(count) + " " + featureStr + "\n";
                    fileN.write(featureStrInst);
        else:
            featureStrInst = str(0) + " " + featureStr + "\n";
            fileN.write(featureStrInst);

        
    def BuildTrainModel(self, bTest):
        global gOutFileList;
        gOutFileList = {};
        gList = gTrainingList;
        if bTest == True:
            gList = gTestingList;
            
        for targetWord in gList:
            binaryAllFeatureValueList = self.GetFeaturesFromTrainingInstance(targetWord);
            self.OutputSVMFormat(binaryAllFeatureValueList, targetWord, bTest);
        
    def GetFeaturesFromTrainingInstance(self, targetWord):
        binaryAllFeatureValueList = [];
        binaryFeatureValueList = self.GetPOSFeatures(targetWord);
        for i in range(len(binaryFeatureValueList)):
            vals = binaryFeatureValueList[i];
            binaryAllFeatureValueList.extend(vals);
            
        binaryFeatureValueList = self.GetSingleWordFeatures(targetWord);
        binaryAllFeatureValueList.extend(binaryFeatureValueList);

        # collocation features
        for i in range(1, TOTAL_COLLOCATION_FEATURE + 1):
            binaryFeatureValueList = self.GetCollocationFeature(targetWord, i);
            binaryAllFeatureValueList.extend(binaryFeatureValueList);

        return binaryAllFeatureValueList;

    def GetPOSFeatures(self, targetWord):
        binaryFeatureValueList = [];
        numPosFeatures = 7;
        halfNumFeatures = math.ceil((numPosFeatures)/2);
        
        for i in range(0, numPosFeatures):
            binaryFeatureValueList.append([0] * len(gPOSTAGS));

        print "Extracting POS word features: ";
        context = targetWord.m_context;
        targetIndex = targetWord.m_ptrWord.m_index;
        bSentStart = False;
        for i in range(0, int(halfNumFeatures)):
            if ((i + targetIndex) >= len(context.m_contextWordList)) | (bSentStart == True):
                break;
            wordCtxt = context.m_contextWordList[targetIndex + i];
            if wordCtxt.m_bSentEnd == True:
                bSentStart = True;
            if not wordCtxt.m_wordPos in gPOSTAGS:
                print "tag NOT FOUND: ", wordCtxt.m_wordPos;
                continue;
            binaryFeatureValueList[i + int(halfNumFeatures) - 1][gPOSTAGS[wordCtxt.m_wordPos] - 1] = 1;

        for i in range(1, int(halfNumFeatures)):
            x = -i;
            if ((x + targetIndex) < 0):
                break;
            wordCtxt = context.m_contextWordList[targetIndex + x];
            if wordCtxt.m_bSentEnd == True:
                break;
            if not wordCtxt.m_wordPos in gPOSTAGS:
                print "tag NOT FOUND: ", wordCtxt.m_wordPos;
                continue;
            binaryFeatureValueList[x + int(halfNumFeatures) - 1][gPOSTAGS[wordCtxt.m_wordPos] - 1] = 1;

        print "Extracted POS word features: ";
        return binaryFeatureValueList;
        
    def GetSingleWordFeatures(self, targetWord):
        binaryFeatureValueList = [];
        for featureWord in self.m_SingleWordList[targetWord.m_wordBase][targetWord.m_wordPos]:
            if featureWord in targetWord.m_singleWordFeatures:
                binaryFeatureValueList.append(1);
            else:
                binaryFeatureValueList.append(0);
        return binaryFeatureValueList;

    def GetCollocationFeature(self, targetWord, collocationTypeIndex):
        binaryFeatureValueList = [];
        if (not self.m_CollocationList[collocationTypeIndex][targetWord.m_wordBase]) | (not self.m_CollocationList[collocationTypeIndex][targetWord.m_wordBase][targetWord.m_wordPos]):
            return [];    
        for featurePhrase in self.m_CollocationList[collocationTypeIndex][targetWord.m_wordBase][targetWord.m_wordPos]:
            if featurePhrase in targetWord.m_collocationFeatures[collocationTypeIndex]:
                binaryFeatureValueList.append(1);
            else:
                binaryFeatureValueList.append(0);
        return binaryFeatureValueList;
    
def ExtractCollocationFeatures(start, end, collocationTypeIndex, bTest):
    global gTrainingList;
    global gCollocationFeatures;
    global gTestingList;

    gList = gTrainingList;
    if bTest == True:
        gList = gTestingList;
    
    #extract all collocations
    print "Extracting collocation type: ", collocationTypeIndex;
    for targetWord in gList:

        collocationFeature = targetWord.m_collocationFeatures[collocationTypeIndex];

        context = targetWord.m_context;
        targetIndex = targetWord.m_ptrWord.m_index;
        collocationList = [];
        bSkip = False;


        if start > 0:
            for i in range(targetIndex +1, start + targetIndex +1):
                if i < len(context.m_contextWordList):
                    if context.m_contextWordList[i].m_bSentEnd == True:
                      bSkip = True;
                      break;
        if end < 0:
            for i in range(targetIndex + end, targetIndex -1):
                if i >= 0:
                    if context.m_contextWordList[i].m_bSentEnd == True:
                      bSkip = True;
                      break;

        if bSkip == True:
            continue;
        
        count = 0;
        bFirstHalf = True;
        bNowNulls = False;
        for i in range(start, end+1):
            index = targetIndex + i;
            if index < 0 | index >= len(context.m_contextWordList):
                continue;

            if i > 0:
                bFirstHalf = False;
            
            if (len(context.m_contextWordList) > index) & (index >= 0):    
              wordCtxt = context.m_contextWordList[index];
            word = wordCtxt.m_word;

            if bNowNulls == True:
                continue;
            
            if wordCtxt.m_bSentEnd == True:
                if bFirstHalf == True:
                    del collocationList[:];
                else:
                    bNowNulls = True;
                continue;

            collocationList.append(word);

        # list empty
        if not collocationList:
            continue;

        collocationStr = ' '.join(collocationList);

        if collocationStr in collocationFeature:
            collocationFeature[collocationStr] = collocationFeature[collocationStr] + 1;
        else:
            collocationFeature[collocationStr] = 1;

        # add also in training feature
        if bTest == False:
            featureCollocationList = gTraining.m_CollocationList[collocationTypeIndex];
            if not targetWord.m_wordBase in featureCollocationList:
                featureCollocationList[targetWord.m_wordBase] = {};
            if not targetWord.m_wordPos in featureCollocationList[targetWord.m_wordBase]:
                featureCollocationList[targetWord.m_wordBase][targetWord.m_wordPos] = {};
           
            if collocationStr in featureCollocationList[targetWord.m_wordBase][targetWord.m_wordPos]:
                featureCollocationList[targetWord.m_wordBase][targetWord.m_wordPos][collocationStr] = featureCollocationList[targetWord.m_wordBase][targetWord.m_wordPos][collocationStr] + 1;
            else:
                featureCollocationList[targetWord.m_wordBase][targetWord.m_wordPos][collocationStr] = 1;

    print "Extracted collocation type: ", collocationTypeIndex;
            

def ExtractFeatures(bTest):
    global gTrainingList;
    global gTestingList;
    global gSingleWordFeature;

    gList = gTrainingList;
    if bTest == True:
        gList = gTestingList;

    #extract collocation strs
    ExtractCollocationFeatures(-1, -1, 1, bTest);
    ExtractCollocationFeatures(1, 1, 2, bTest);
    ExtractCollocationFeatures(-2, -2, 3, bTest);
    ExtractCollocationFeatures(2, 2, 4, bTest);
    ExtractCollocationFeatures(-2, -1, 5, bTest);
    ExtractCollocationFeatures(-1, 1, 6, bTest);
    ExtractCollocationFeatures(1, 2, 7, bTest);
    ExtractCollocationFeatures(-3, -1, 8, bTest);
    ExtractCollocationFeatures(-2, 1, 9, bTest);
    ExtractCollocationFeatures(-1, 2, 10, bTest);
    ExtractCollocationFeatures(1, 3, 11, bTest);

    print "Extracting single word features: ";
    # extract all single-words
    for targetWord in gList:
        context = targetWord.m_context;
        for contextWord in context.m_contextWordList:
            word = contextWord.m_word;
            if (word in nltk.corpus.stopwords.words('english')) | (not word.isalpha()):
                continue;

            if word in targetWord.m_singleWordFeatures:
                targetWord.m_singleWordFeatures[word] = targetWord.m_singleWordFeatures[word] + 1;
            else:
                targetWord.m_singleWordFeatures[word] = 1;

            if bTest == False:
                # add in training feature
                if not targetWord.m_wordBase in gTraining.m_SingleWordList:
                    gTraining.m_SingleWordList[targetWord.m_wordBase] = {};
                if not targetWord.m_wordPos in gTraining.m_SingleWordList[targetWord.m_wordBase]:
                    gTraining.m_SingleWordList[targetWord.m_wordBase][targetWord.m_wordPos] = {};

                if word in gTraining.m_SingleWordList[targetWord.m_wordBase][targetWord.m_wordPos]:
                    gTraining.m_SingleWordList[targetWord.m_wordBase][targetWord.m_wordPos][word] = gTraining.m_SingleWordList[targetWord.m_wordBase][targetWord.m_wordPos][word] + 1;
                else:
                    gTraining.m_SingleWordList[targetWord.m_wordBase][targetWord.m_wordPos][word] = 1;

    print "Extracted single word features: ";


def ReadTrainingData(bTest):
    global fT;
    count = 1;
    f = gTrainingFile;
    if bTest == True:
        f = gTestingFile;
    fTrain = open(f, 'r')

    #fT = open('TestWords.txt', 'w');

    for line in fTrain:
        print count;
        if line is None:
            print 'Reading Training: line None';
            continue;

        ParseTrainingData(line, bTest);
        count = count + 1;
        

def ParseTrainingData(data, bTest):
    global gSent_detector;
    global gTrainingList;
    global gTestingList;

    gList = gTrainingList;
    if bTest == True:
        gList = gTestingList;

    
    data = data.lower();    # lowercase
    [firstWord, rest] = data.split(' ', 1);

    # firstWord contains base-form word and its pos tag
    [targetWordBase, targetWordPos] = firstWord.split('.', 1);


    # get senses and context
    [senses, context] = rest.split('@', 1);
    senses = senses.strip();
    sensesList = senses.split(' ');
    sensesList = [int(sense) for sense in sensesList];  # convert to integers

    #code to put all words in file
    #fT.write(firstWord + " " + str(len(sensesList)) + '\n');

    # get target-word
    [lCntxt, tWord, rCntxt] = context.split('@', 2);

    # store the data in word class
    targetWord = TargetWord();
    targetWord.m_line = data;
    targetWord.m_word = tWord;
    targetWord.m_wordBase = targetWordBase;
    targetWord.m_wordPos = targetWordPos;
    targetWord.m_senses = sensesList;
#    gTrainingList.append(targetWord);

    wordContext = TargetWordContext();
    wordContext.m_contextText = context;

    sentences = gSent_detector.tokenize(context);

    countTotal = 0;
    for sent in sentences:
#        print 'Sentence: ', sent;

        p = re.compile('@(([a-zA-Z])*)@');
        sent = p.sub(r'X\1X', sent);

        wordList = nltk.tokenize.word_tokenize(sent);
        posList = nltk.pos_tag(wordList);

        count = 0;        
        for word in wordList:
            contextWord = ContextWord();
            contextWord.m_wordPos = posList[count][1];
 
            # see if this is the target word
            if (len(word) >= 2) & (word[0] == 'X') & (word[len(word) -1] == 'X'):
                word = word.strip('X');
                targetWord.m_ptrWord = contextWord;

            contextWord.m_word = word;
            
            stemmedWord = nltk.stem.porter.PorterStemmer().stem_word(word);
            contextWord.m_stemWord = stemmedWord;
            contextWord.m_targetWordContext = wordContext;
            contextWord.m_index = countTotal;
            contextWord.m_sentIndex = count;
            count = count + 1;
            if count == len(wordList):
                contextWord.m_bSentEnd = True;

            wordContext.m_contextWordList.append(contextWord);
            countTotal = countTotal + 1;

    targetWord.m_context = wordContext;  
    gList.append(targetWord);        
    
#    print 'Training Example: ', ' Target-Word Base: ', targetWordBase, ' Target-Word POS: ', targetWordPos, ' senses: ', senses, ' context: ', context;

def ReadDictionary():
    global gDictionary;
    doc = xml.dom.minidom.parse(gDictionaryFile);
    
    for root in doc.getElementsByTagName("dictmap"):
        for node in root.getElementsByTagName("lexelt"):
            item = str(node.getAttribute("item"));
            [wordBase, wordPos] = item.split('.', 1);
            bunch = Bunch(word="",posDict={});
            bunch.word = wordBase;
            senseList = [];

            S = node.getElementsByTagName("sense")
            for node2 in S:
                id = str(node2.getAttribute("id"));
                synset = str(node2.getAttribute("synset"));
                gloss = str(node2.getAttribute("gloss"));

                synsetList = synset.split(' ');
                glossList = gloss.split(';');

                sense = Sense();
                sense.m_senseId = id;
                sense.m_synsetsList = synsetList;
                sense.m_glossesList = glossList;

                senseList.append(sense);

            bunch.posDict[wordPos] = senseList;
            gDictionary.append(bunch);
    
def Init():
    global gSent_detector;
    global gCollocationFeatures;
    global gTraining;

    gTraining = Training();
    gSent_detector = nltk.data.load('tokenizers/punkt/english.pickle');

def SegmentToTestTrain():
    global gTrainingList;
    mapf = {};
    fTest = open('validation.txt', 'w');
    fTrain = open('train.txt', 'w');

    for targetWord in gTrainingList:
        if not mapf.has_key(targetWord.m_wordBase):
            mapf[targetWord.m_wordBase] = [];
        mapf[targetWord.m_wordBase].append(targetWord.m_line);

    for word in mapf:
        linesTest = [];
        lines = mapf[word];
        sz = len(lines);
        szTest = math.ceil(sz * 0.2);

        for i in range(0, szTest):
            sx = len(lines);
            r = random.randint(0, sx - 1);
            linesTest.append(lines[r]);
            fTest.write(lines[r]);
            lines.pop(r);

        for i in range(len(lines)):
            fTrain.write(lines[i]);

    fTest.close();
    fTrain.close();

            
                
def main():
    global gTraining;
    Init();
    #ReadDictionary();
    ReadTrainingData(False);
    #SegmentToTestTrain();
    ExtractFeatures(False);
    gTraining.BuildTrainModel(False);

    #pdb.set_trace();
    #Testing
    print 'Now Testing';
    ReadTrainingData(True);
    #SegmentToTestTrain();
    ExtractFeatures(True);
    gTraining.BuildTrainModel(True);
  
if __name__ == "__main__":
  main()
