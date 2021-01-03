# A Dependency Parsing version of the PennTreeBank Dataset
A Dependency Parsing version of the PennTreeBank for the WSJ and Brown corpora. Includes preprocessing based on the LTH tool and a Dataset class for use in PyTorch.

Please see the attached notebook for the full pipeline and an example of how to use the provided Dataset class.

This also precalculates the Dependency Graph for each sample as Numpy arrays.

As the original PennTreeBank dataset uses Constituency Parsing, we used the LTH tool from http://nlp.cs.lth.se/software/treebank_converter/. This is a Java based application, however our script uses it by leveraging the `subprocess` Python library.

### Why is this repository necessary?

It is a convenience for those who wish to use the WSJ and Brown corpora using Dependendency Parsing and have no idea how to convert the files (like me at the start). The main part of this (the conversion) is the work of the people who developed LTH.

### Why does this work only for the WSJ and Brown corpora?

Because LTH only works for those datasets. I believe there are no annotated trees for the rest of the datasets as well.

The tool works on .mrg files from the `mrg` directory in the Treebank dataset. It will take a file looking something like this:

```

( (S 
    (NP-SBJ 
      (NP (NNP Pierre) (NNP Vinken) )
      (, ,) 
      (ADJP 
        (NP (CD 61) (NNS years) )
        (JJ old) )
      (, ,) )
    (VP (MD will) 
      (VP (VB join) 
        (NP (DT the) (NN board) )
        (PP-CLR (IN as) 
          (NP (DT a) (JJ nonexecutive) (NN director) ))
        (NP-TMP (NNP Nov.) (CD 29) )))
    (. .) ))
( (S 
    (NP-SBJ (NNP Mr.) (NNP Vinken) )
    (VP (VBZ is) 
      (NP-PRD 
        (NP (NN chairman) )
        (PP (IN of) 
          (NP 
            (NP (NNP Elsevier) (NNP N.V.) )
            (, ,) 
            (NP (DT the) (NNP Dutch) (VBG publishing) (NN group) )))))
    (. .) ))

```

And convert it to something like this:

```
1	Pierre	_	NNP	_	_	2	NAME	_	_
2	Vinken	_	NNP	_	_	8	SBJ	_	_
3	,	_	,	_	_	2	P	_	_
4	61	_	CD	_	_	5	NMOD	_	_
5	years	_	NNS	_	_	6	AMOD	_	_
6	old	_	JJ	_	_	2	APPO	_	_
7	,	_	,	_	_	2	P	_	_
8	will	_	MD	_	_	0	ROOT	_	_
9	join	_	VB	_	_	8	VC	_	_
10	the	_	DT	_	_	11	NMOD	_	_
11	board	_	NN	_	_	9	OBJ	_	_
12	as	_	IN	_	_	9	ADV	_	_
13	a	_	DT	_	_	15	NMOD	_	_
14	nonexecutive	_	JJ	_	_	15	NMOD	_	_
15	director	_	NN	_	_	12	PMOD	_	_
16	Nov.	_	NNP	_	_	9	TMP	_	_
17	29	_	CD	_	_	16	NMOD	_	_
18	.	_	.	_	_	8	P	_	_

1	Mr.	_	NNP	_	_	2	TITLE	_	_
2	Vinken	_	NNP	_	_	3	SBJ	_	_
3	is	_	VBZ	_	_	0	ROOT	_	_
4	chairman	_	NN	_	_	3	PRD	_	_
5	of	_	IN	_	_	4	NMOD	_	_
6	Elsevier	_	NNP	_	_	5	PMOD	_	_
7	N.V.	_	NNP	_	_	6	POSTHON	_	_
8	,	_	,	_	_	6	P	_	_
9	the	_	DT	_	_	12	NMOD	_	_
10	Dutch	_	NNP	_	_	12	NMOD	_	_
11	publishing	_	VBG	_	_	12	NMOD	_	_
12	group	_	NN	_	_	6	APPO	_	_
13	.	_	.	_	_	3	P	_	_
```
Files in this format will be saved as .pd files.

The code in this repository will also create two extra files for each file in the dataset:

* *.txt*: These files have the plain text version of the sentences. Sample output:
```
Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .
Mr. Vinken is chairman of Elsevier N.V. , the Dutch publishing group .

```
* *.npy*: These files contain numpy arrays for the Dependency Graphs for the associated samples. Arrays are of size $n \times n$
, where $n$ is the number of parts (words, punctuation, etc.) in the sentence. Each row represents a sentence part and each column denotes whether the corresponding sentence part is a parent of the current sentence part.

## Dataset

The code for the Dataset class for PyTorch use is provided both on the sample notebook as well as in `penndb.py`.

In its base form, the dataset will return both a list with the sentence parts and also the associated dependency graph.

However, the dataset takes in an optional parameter `tokenizer`, which takes a `PreTrainedTokenizer` from the `transformers` library. This will tokenize the sentences and will be returned in addition to the sentence parts and the graph.

Example:

```python
sentence, matrix, tokens = data[0]

print("The original sentence is:")
print(" ".join(sentence))
print("The decoded sentence is:")
print(tokenizer.decode(tokens))
```
Output:

```
The original sentence is:
Jim Pattison Industries Ltd. , one of a group of closely held companies owned by entrepreneur James Pattison , said it `` intends to seek control '' of 30%-owned Innopac Inc. , a Toronto packaging concern .
The decoded sentence is:
[CLS] Jim Pattison Industries Ltd., one of a group of closely held companies owned by entrepreneur James Pattison, said it ` ` intends to seek control'' of 30 % - owned Innopac Inc., a Toronto packaging concern. [SEP]
```
