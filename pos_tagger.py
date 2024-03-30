import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import sys

class SimpleMLEModel(nn.Module):
    def __init__(self, vocab_size, tagset_size,sen_size, embedding_dim, hidden_dim):
        super(SimpleMLEModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(embedding_dim*sen_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, tagset_size)
        self.relu = nn.ReLU()

    def forward(self, sentence):
      embeds = self.embedding(sentence)
      embeds = torch.flatten(embeds, start_dim=1)
      # print(embeds.shape)
      out = self.linear1(embeds)
      out = self.relu(out)
      out = self.linear2(out)
      return out

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(VanillaRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Convert input indices to embeddings
        embeds = self.embedding(x)

        # Pack the sequences
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)

        # Initialize hidden state
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        # Forward pass through RNN
        packed_out, hidden = self.rnn_cell(packed_embeds, hidden)

        # Unpack the sequences
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Output from the fully connected layer
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        return out

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

word_to_ix = {'what': 0,
 'is': 1,
 'the': 2,
 'cost': 3,
 'of': 4,
 'a': 5,
 'round': 6,
 'trip': 7,
 'flight': 8,
 'from': 9,
 'pittsburgh': 10,
 'to': 11,
 'atlanta': 12,
 'on': 13,
 'april': 14,
 'twenty': 15,
 'fifth': 16,
 'and': 17,
 'returning': 18,
 'may': 19,
 'sixth': 20,
 'now': 21,
 'i': 22,
 'need': 23,
 'leaving': 24,
 'fort': 25,
 'worth': 26,
 'arriving': 27,
 'in': 28,
 'denver': 29,
 'no': 30,
 'later': 31,
 'than': 32,
 '2': 33,
 'pm': 34,
 'next': 35,
 'monday': 36,
 'fly': 37,
 'kansas': 38,
 'city': 39,
 'chicago': 40,
 'wednesday': 41,
 'following': 42,
 'day': 43,
 'meaning': 44,
 'meal': 45,
 'code': 46,
 's': 47,
 'show': 48,
 'me': 49,
 'all': 50,
 'flights': 51,
 'which': 52,
 'serve': 53,
 'for': 54,
 'after': 55,
 'tomorrow': 56,
 'us': 57,
 'air': 58,
 'list': 59,
 'nonstop': 60,
 'early': 61,
 'tuesday': 62,
 'morning': 63,
 'dallas': 64,
 'st.': 65,
 'petersburg': 66,
 'toronto': 67,
 'that': 68,
 'arrive': 69,
 'listing': 70,
 'new': 71,
 'york': 72,
 'montreal': 73,
 'canada': 74,
 'departing': 75,
 'thursday': 76,
 'american': 77,
 'airlines': 78,
 'ontario': 79,
 'with': 80,
 'stopover': 81,
 'louis': 82,
 'ground': 83,
 'transportation': 84,
 'houston': 85,
 'afternoon': 86,
 'schedule': 87,
 'philadelphia': 88,
 'san': 89,
 'francisco': 90,
 'evening': 91,
 'diego': 92,
 'layover': 93,
 'washington': 94,
 'dc': 95,
 'are': 96,
 'there': 97,
 'any': 98,
 'boston': 99,
 'stop': 100,
 'restrictions': 101,
 'cheapest': 102,
 'one': 103,
 'way': 104,
 'fare': 105,
 'between': 106,
 'oakland': 107,
 'airfare': 108,
 'dollars': 109,
 "'s": 110,
 'restriction': 111,
 'california': 112,
 'airports': 113,
 'available': 114,
 'texas': 115,
 'airport': 116,
 'nevada': 117,
 'arizona': 118,
 'las': 119,
 'vegas': 120,
 'burbank': 121,
 'saturday': 122,
 'two': 123,
 'how': 124,
 'many': 125,
 'going': 126,
 'july': 127,
 'seventh': 128,
 'would': 129,
 'like': 130,
 'an': 131,
 'february': 132,
 'eighth': 133,
 'before': 134,
 '9': 135,
 'am': 136,
 'second': 137,
 'late': 138,
 'okay': 139,
 'june': 140,
 'first': 141,
 "'d": 142,
 'go': 143,
 'phoenix': 144,
 'detroit': 145,
 'milwaukee': 146,
 'indianapolis': 147,
 'does': 148,
 'stand': 149,
 'as': 150,
 'baltimore': 151,
 '1115': 152,
 '1245': 153,
 'miami': 154,
 'daily': 155,
 '8': 156,
 'airline': 157,
 '5': 158,
 'leave': 159,
 'charlotte': 160,
 'north': 161,
 'carolina': 162,
 '4': 163,
 'find': 164,
 'newark': 165,
 'jersey': 166,
 'cleveland': 167,
 'ohio': 168,
 'do': 169,
 'you': 170,
 'have': 171,
 'connect': 172,
 'international': 173,
 'minneapolis': 174,
 'rental': 175,
 'cars': 176,
 "'ll": 177,
 'rent': 178,
 'car': 179,
 'near': 180,
 'can': 181,
 'give': 182,
 'information': 183,
 'downtown': 184,
 'economy': 185,
 'class': 186,
 'fares': 187,
 'december': 188,
 'sixteenth': 189,
 'codes': 190,
 'coach': 191,
 'night': 192,
 'service': 193,
 'november': 194,
 'twelfth': 195,
 'eleventh': 196,
 'want': 197,
 'know': 198,
 'or': 199,
 '1': 200,
 "o'clock": 201,
 '3': 202,
 '6': 203,
 '10': 204,
 'august': 205,
 'display': 206,
 'depart': 207,
 'please': 208,
 'served': 209,
 'tower': 210,
 'types': 211,
 'meals': 212,
 'my': 213,
 'options': 214,
 'get': 215,
 'bwi': 216,
 'eastern': 217,
 '210': 218,
 'delta': 219,
 '852': 220,
 'latest': 221,
 'return': 222,
 'same': 223,
 'back': 224,
 'most': 225,
 'hours': 226,
 'take': 227,
 'so': 228,
 'when': 229,
 'will': 230,
 'maximum': 231,
 'amount': 232,
 'time': 233,
 'still': 234,
 'earliest': 235,
 'departure': 236,
 'be': 237,
 'travel': 238,
 'at': 239,
 'around': 240,
 '7': 241,
 'lastest': 242,
 'but': 243,
 'possible': 244,
 'only': 245,
 'weekdays': 246,
 'los': 247,
 'angeles': 248,
 'ten': 249,
 'people': 250,
 'during': 251,
 'week': 252,
 'days': 253,
 'out': 254,
 'arrives': 255,
 'salt': 256,
 'lake': 257,
 'cincinnati': 258,
 'area': 259,
 'explain': 260,
 'ap': 261,
 '57': 262,
 'mean': 263,
 '80': 264,
 'twa': 265,
 'has': 266,
 'stops': 267,
 'friday': 268,
 'number': 269,
 'book': 270,
 'least': 271,
 '813': 272,
 'goes': 273,
 'through': 274,
 'without': 275,
 'stopping': 276,
 'florida': 277,
 'tell': 278,
 'about': 279,
 'by': 280,
 'memphis': 281,
 'tennessee': 282,
 'noon': 283,
 '530': 284,
 'love': 285,
 'field': 286,
 'united': 287,
 'la': 288,
 'guardia': 289,
 'jfk': 290,
 'mco': 291,
 'sfo': 292,
 '1991': 293,
 'orlando': 294,
 'lowest': 295,
 'dfw': 296,
 'ticket': 297,
 'logan': 298,
 'march': 299,
 'numbers': 300,
 'expensive': 301,
 'continental': 302,
 'leaves': 303,
 'seattle': 304,
 'columbus': 305,
 'minnesota': 306,
 'those': 307,
 'via': 308,
 'rentals': 309,
 'sunday': 310,
 'rates': 311,
 'costs': 312,
 'limousine': 313,
 'taxi': 314,
 'ap80': 315,
 'ninth': 316,
 '12': 317,
 'america': 318,
 'west': 319,
 'could': 320,
 'fifteenth': 321,
 'serves': 322,
 'dinner': 323,
 'provided': 324,
 'cities': 325,
 'where': 326,
 'canadian': 327,
 'other': 328,
 'northwest': 329,
 'general': 330,
 'mitchell': 331,
 'both': 332,
 'nationair': 333,
 'midwest': 334,
 'express': 335,
 'flies': 336,
 'flying': 337,
 'into': 338,
 'much': 339,
 'price': 340,
 'it': 341,
 'tacoma': 342,
 'anywhere': 343,
 'midnight': 344,
 'january': 345,
 '1992': 346,
 'not': 347,
 '300': 348,
 'tenth': 349,
 '1993': 350,
 'october': 351,
 '1994': 352,
 'smallest': 353,
 'passengers': 354,
 'thirty': 355,
 'third': 356,
 'arrival': 357,
 'schedules': 358,
 'times': 359,
 'your': 360,
 '269': 361,
 'westchester': 362,
 'county': 363,
 'right': 364,
 'september': 365,
 'twentieth': 366,
 'f28': 367,
 'their': 368,
 'prices': 369,
 '1039': 370,
 'less': 371,
 '1100': 372,
 'nashville': 373,
 'again': 374,
 'repeat': 375,
 'make': 376,
 'ord': 377,
 'ewr': 378,
 'dca': 379,
 'long': 380,
 'distance': 381,
 'far': 382,
 'paul': 383,
 'miles': 384,
 'name': 385,
 'serviced': 386,
 'tampa': 387,
 'names': 388,
 'describe': 389,
 'nineteenth': 390,
 'seating': 391,
 'capacity': 392,
 'fourteenth': 393,
 'aircraft': 394,
 'plane': 395,
 'eight': 396,
 'sixteen': 397,
 'departures': 398,
 'seventeenth': 399,
 'arrivals': 400,
 'type': 401,
 'more': 402,
 'business': 403,
 'total': 404,
 'turboprop': 405,
 'land': 406,
 'various': 407,
 'dulles': 408,
 'boeing': 409,
 '767': 410,
 '466': 411,
 'under': 412,
 '932': 413,
 '1000': 414,
 '200': 415,
 'along': 416,
 'each': 417,
 'fit': 418,
 '72s': 419,
 'airplane': 420,
 'hold': 421,
 '733': 422,
 'airplanes': 423,
 'uses': 424,
 '73s': 425,
 'seats': 426,
 'm80': 427,
 'capacities': 428,
 '757': 429,
 'planes': 430,
 'd9s': 431,
 'thrift': 432,
 'see': 433,
 'thirtieth': 434,
 '505': 435,
 'connecting': 436,
 'also': 437,
 'making': 438,
 "'m": 439,
 'looking': 440,
 'makes': 441,
 'yes': 442,
 'breakfast': 443,
 'direct': 444,
 'departs': 445,
 'provide': 446,
 'used': 447,
 'connections': 448,
 'if': 449,
 'either': 450,
 'beach': 451,
 'then': 452,
 'mornings': 453,
 'four': 454,
 'thank': 455,
 'using': 456,
 'well': 457,
 'colorado': 458,
 'fourth': 459,
 'who': 460,
 'sure': 461,
 'determine': 462,
 'use': 463,
 'lufthansa': 464,
 'eighteenth': 465,
 'f': 466,
 'today': 467,
 'booking': 468,
 'classes': 469,
 'yn': 470,
 'j31': 471,
 'different': 472,
 'connection': 473,
 'last': 474,
 'aa': 475,
 'services': 476,
 'jose': 477,
 'too': 478,
 'georgia': 479,
 'pennsylvania': 480,
 'utah': 481,
 'missouri': 482,
 'interested': 483,
 'shortest': 484,
 'quebec': 485,
 'michigan': 486,
 'indiana': 487,
 'this': 488,
 'wednesdays': 489,
 'tickets': 490,
 'great': 491,
 'let': 492,
 'takeoffs': 493,
 'landings': 494,
 'offer': 495,
 'transport': 496,
 'kind': 497,
 'hi': 498,
 'coming': 499,
 'soon': 500,
 'up': 501,
 'y': 502,
 'm': 503,
 'difference': 504,
 'q': 505,
 'qo': 506,
 'qw': 507,
 'qx': 508,
 'fn': 509,
 'h': 510,
 'offers': 511,
 'serving': 512,
 'trying': 513,
 'include': 514,
 'ua': 515,
 '270': 516,
 '747': 517,
 'very': 518,
 'three': 519,
 '727': 520,
 'dc10': 521,
 'abbreviation': 522,
 '296': 523,
 'should': 524,
 'lunch': 525,
 '343': 526,
 '838': 527,
 '1110': 528,
 'sometime': 529,
 'some': 530,
 'saturdays': 531,
 'destination': 532,
 'over': 533,
 '1222': 534,
 '281': 535,
 'dl': 536,
 '201': 537,
 '21': 538,
 '825': 539,
 '555': 540,
 '1500': 541,
 '217': 542,
 '3724': 543,
 '1291': 544,
 'co': 545,
 'ea': 546,
 'traveling': 547,
 'these': 548,
 'takeoff': 549,
 'close': 550,
 '230': 551,
 'nonstops': 552,
 'thursdays': 553,
 "'re": 554,
 '1200': 555,
 'reservation': 556,
 'lives': 557,
 '11': 558,
 'we': 559,
 'southwest': 560,
 '630': 561,
 'hp': 562,
 'define': 563,
 'ff': 564,
 'stands': 565,
 'ac': 566,
 '718': 567,
 'arrangements': 568,
 'originating': 569,
 'choices': 570,
 '1145': 571,
 'rate': 572,
 'trips': 573,
 'stopovers': 574,
 'starting': 575,
 'reservations': 576,
 'six': 577,
 '1700': 578,
 '2100': 579,
 'wish': 580,
 'look': 581,
 'transcontinental': 582,
 'month': 583,
 'help': 584,
 '720': 585,
 '934': 586,
 'heading': 587,
 '430': 588,
 'weekday': 589,
 'cheap': 590,
 'live': 591,
 'highest': 592,
 'noontime': 593,
 'tuesdays': 594,
 'alaska': 595,
 'arrange': 596,
 'plan': 597,
 'hello': 598,
 'requesting': 599,
 '<UNK>': 600,
 'SOS': 601,
 'EOS': 602}

tag_to_ix = {
              'PRON': 0,
              'AUX': 1,
              'DET': 2,
              'NOUN': 3,
              'ADP': 4,
              'PROPN': 5,
              'VERB': 6,
              'NUM': 7,
              'ADJ': 8,
              'CCONJ': 9,
              'ADV': 10,
              'PART': 11,
              'INTJ': 12
            }

unknown  = "<UNK>"

def tets_on_fnn(test_data,model,batch_size,word_to_ix,tag_to_ix):
  pred_label = []
  actual_label = []
  with torch.no_grad():
    for i in range(0, len(test_data), batch_size):
      sentences = test_data[i:i+batch_size]
 
      input_seqs = [torch.tensor([word_to_ix[word] for word in sentence], dtype=torch.long) for sentence in sentences]
      

      padded_batch = pad_sequence(input_seqs, batch_first=True)
      outputs = model(padded_batch)

      max_indices = torch.argmax(outputs, dim=1)
      rev = {0: 'PRON', 1: 'AUX', 2: 'DET', 3: 'NOUN', 4: 'ADP', 5: 'PROPN', 6: 'VERB', 7: 'NUM', 8: 'ADJ', 9: 'CCONJ', 10: 'ADV', 11: 'PART', 12: 'INTJ'}
      # print(max_indices)
      pred_label.extend([ rev[i] for i in max_indices.tolist() ])
      

  return pred_label

def prepare_windows(sentence,n,p):
  train_data = []
  k = len(sentence)
  temp = [ 'SOS' for i in range(p)]
  temp.extend(sentence)
  temp.extend(["EOS" for i in range(n)])

  for i in range(k):
    train_data.append(temp[i:(i + n+p+1)])

  return train_data

def tets_on_rnn(test_data,model,batch_size,word_to_ix,tag_to_ix):
  pred_label = []
  actual_label = []
  test_data = [test_data]
  with torch.no_grad():
    for i in range(0, len(test_data), batch_size):
      sentences = test_data[i:i+batch_size]
      

      input_seqs = [torch.tensor([word_to_ix[word] for word in sentence], dtype=torch.long) for sentence in sentences]
      
      input_seqs_padded = pad_sequence(input_seqs, batch_first=True)
      lengths = [len(seq) for seq in input_seqs]

      outputs = model(input_seqs_padded, lengths)

      _, predicted_tags = torch.max(outputs, 2)

      rev = {0: 'PRON', 1: 'AUX', 2: 'DET', 3: 'NOUN', 4: 'ADP', 5: 'PROPN', 6: 'VERB', 7: 'NUM', 8: 'ADJ', 9: 'CCONJ', 10: 'ADV', 11: 'PART', 12: 'INTJ'}
      # print(max_indices)
      

      for p in range(len(predicted_tags)):
        x = predicted_tags[p][:lengths[p]]
        pred_label.extend([ rev[i] for i in x.tolist() ])

  return pred_label

arg1 = sys.argv[1] 

sentence = input()
sentence = sentence.split()


if arg1 == '-f' :
    vocab_size = len(word_to_ix)
    tagset_size = len(tag_to_ix)
    embedding_dim = 25
    hidden_dim = 25
    n=1
    p=1

    model_path = "./fnn_window_new1.pth"
    model3 = SimpleMLEModel(vocab_size,tagset_size,n+p+1, embedding_dim, hidden_dim)
    model3.load_state_dict(torch.load(model_path))
    model3.eval()

    data = prepare_windows(sentence,1,1)

    for row in range(len(data)):
        for word in range(len(data[row])):
            if data[row][word] not in word_to_ix.keys():
                data[row][word] = unknown

    ans = tets_on_fnn(data,model3,1,word_to_ix,tag_to_ix)

    for i in range(len(ans)):
        print(sentence[i],ans[i])

else :

    vocab_size = 932
    tagset_size = len(tag_to_ix)
    embedding_dim = 50
    hidden_dim = 20

    model_path = './vanilla_rnn_model_1.pth'
    model3 = VanillaRNN(vocab_size, embedding_dim, hidden_dim, tagset_size)
    model3.load_state_dict(torch.load(model_path))
    model3.eval()

    for word in range(len(sentence)):
        if sentence[word] not in word_to_ix.keys():
            sentence[word] = "EOS"

    ans = tets_on_rnn(sentence,model3,1,word_to_ix,tag_to_ix)

    for i in range(len(ans)):
        print(sentence[i],ans[i])