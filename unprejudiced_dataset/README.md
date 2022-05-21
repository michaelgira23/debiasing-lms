# "Unprejudiced Dataset"

While no dataset is perfect, our aim is to create a corpus that contains an even distribution of demographic representation, so these prejudices are less prevalent. It contains text from these projects:

-   [StereoSet](https://github.com/moinnadeem/StereoSet)
-   [WinoBias](https://github.com/uclanlp/corefBias)
-   [CrowS-Pairs](https://github.com/nyu-mll/crows-pairs)

## Processing the WinoBias dataset

### Removing the brackets

Replace RegEx

```Re
\[([^\]]+)\]
```

with

```
$1
```

### Removing the numbering

Replace RegEx (note the space at the end)

```Re
^\d+
```

with (nothing)

```

```
