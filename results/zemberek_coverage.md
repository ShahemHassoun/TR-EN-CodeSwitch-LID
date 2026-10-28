# Zemberek coverage on the train split

Recognition is measured over token occurrences labeled `TR` in the train split. A token is recognized if Zemberek returns at least one non-runtime, non-unknown analysis that is not solely a proper-noun analysis. No validation or test labels are used.

| Turkish characters in token | Recognition rate |
|---|---:|
| With ç, ğ, ı, ö, ş, or ü | 97.4% (2895/2972) |
| Without those characters | 83.1% (5806/6990) |
