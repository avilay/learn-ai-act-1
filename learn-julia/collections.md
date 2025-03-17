# Functions on Collections

| Similar Functions                                            | Notes                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `isempty`, `empty!`                                          |                                                              |
| `length`, `size`, `ndims`, `eachindex`                       |                                                              |
| `in`, $\in$, $\notin$, $\ni$, `insorted`, `contains`, `occursin` |                                                              |
| `indexin`, `sortperm`, `findfirst`                           | `indexin(ary1, ary2)` will take each element of `ary1` and returns its index (first occurance) in `ary2`. |
| `unique`, `unique!`, `allunique`,  `unique(f, itr)`          |                                                              |
| `reduce`, `foldl`, `foldr`, `sum`, `mapreduce`, `count`, `sum!`, `prod`, `prod!`, `cumprod`, `mapfoldl`, `mapfoldr`, |                                                              |
| `maximum`, `maximum!`, `extrema`, `findmax`, `argmax`, `minimum`, `minimum!`, `findmin`, `argmin`, `findmax!`, `findmin!` |                                                              |
| `any`, `any!`, `all`, `all!`                                 |                                                              |
| `foreach`, `map`, `map!`, `mapslices`, `zip`                 | Use `foreach` when results of the mapping function are not needed. |
| `first`, `only`, `firstindex`, `last`, `lastindex`, `tail`, `rest`, `front` | Haskell equivalents - `first` $\equiv$ `head`, `front` $\equiv$ `init`, |
| `step`                                                       |                                                              |
| `collect`                                                    |                                                              |
| `filter`, `filter!`                                          |                                                              |
| `replace`, `replace!`,                                       |                                                              |
| `haskey`, `get`, `get!`, `getkey`, `delete!`, `pop!`, `keys`, `values`, `pairs`, `merge`, `mergewith`, `merge!`, `mergewith!`, `size hint!` | These functions are implemented for `Dict` and partially implemented for `Set` and `Array`. |
| `union`, $\cup$, `union!`, `intersect`, $\cap$, `intersect!`, `isdisjoint`, `vcat`, `setdiff`, `setdiff!`, `symdiff`, `symdiff!`, `issubset`, $\sub$, $\subseteq$, $\nsubseteq$, `issetequal` | These functions are implemented for `Set` and partially implemented for `Array`. |
| `push!`,`append!`, `pop!`, `pushfirst!`, `popfirst!`, `popat!`, `delete!`, `deleteat!`, `splice!`, `insert!`, `keepat!`, `resize!`, `prepend!` | These functions are fully implemented for one dimensional arrays aka `Vector`s. |
| `zip`, `enumerate`, `splat`, `drop`, `peel`, `countfrom`, `take`, `take!`, `takewhile`, `dropwhile`, `cycle`, `repeated`, `repeat`, `product`, `flatten`, `partition`, `map`, `filter`, `accumulate`, `reverse`, `only` | These functions are in the `Base.Iterators` package.         |
| `getindex`, `zeros`, `fill`, `ones`, `trues`, `falses`, `fill!`, `similar`, `empty`, `empty!`, `isempty`, `size`, `axes`,`length`, `eachindex`, `sizeof`, `keys`, `broadcast`, `broadcast!`, `@__dot__`, `copy!`, `checkbounds`, `checkindex`, `reshape`, `cat`, | Only array type functions here. Have skipped the tensor functions. |

## String Functions

| Similar Functions                                            | Notes                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `codepoint`, `ncodeunits`, `codeunit`, `codeunits`, `ascii`, `isascii`, `iscntrl`, `isdigit`, `isletter`, `islowercase`, `isuppercase`, `isnumeric`, `isprint`, `ispunct`, `isspace`, `isxdigit` |                                                              |
| `length`, `sizeof`, `textwidth`                              |                                                              |
| `isvalid`                                                    |                                                              |
| `thisind`, `nextind`, `prevind`                              |                                                              |
| `*`, `^`, `repeat`                                           |                                                              |
| `string`, `repr`, `show`, `sprint`                           |                                                              |
| `String`, `transcode`, `@b_str`                              | Similar to `encode` where it converts the string to unicode bytes encoded as utf-8. |
| `SubString`, `findfirst`, `findnext`, `findlast`, `findprev`, `occursin`, `contains`, `reverse`, `startswith`, `endswith`,`occursin`, `in`, `issubset`, `first`, `last` |                                                              |
| `Regex`, `@r_str`, `SubstitutionString`, `@s_str`, `@raw_string`, `match`, `eachmatch`, `keys` |                                                              |
| `@html_str`, `@text_str`                                     |                                                              |
| `isless`, `==`, `cmp`                                        |                                                              |
| `lpad`, `rpad`, `replace`, `split`, `rsplit`, `strip`, `lstrip`, `rstrip`, `join`, `chop`, `chomp` |                                                              |
| `uppercase`, `lowercase`, `titlecase`, `lowercasefirst`, `uppercasefirst` |                                                              |

