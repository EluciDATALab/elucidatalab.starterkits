# ©, 2019, Sirris
# owner: SKLE

"""Tools identify common patterns in strings."""
import pandas as pd
import re
import string
import itertools
from datetime import datetime
import numpy as np


def _test_ts_format(x, ts_format):
    try:
        x = datetime.strptime(x, ts_format)
        return x
    except ValueError:
        return np.nan


def _create_regex_from_pattern_2(x):
    x = x.replace('-', r'\-')
    x = x.replace('.', r'\\.')
    x = x.replace('dd', '([0-2]{1}[0-9]{1}|3[0-1]{1})')
    x = re.sub(r'(?<![h]|[H])MM', '([0][0-9]|1[0-2])', x)
    x = re.sub(r'(?<![h]|[H])mm', '([0][0-9]|1[0-2])', x)
    x = x.replace('yyyy', r'20[0-2]{1}[0-9]{1}')
    x = x.replace('yy', r'[0-2]{1}[0-9]{1}')
    x = x.replace('HH', r'([0-1]{1}[0-9]{1}|2[0-4]{1})')
    x = x.replace('mm', r'([0-5]{1}[0-9]{1})')
    x = re.sub('(?<![y]|[Y]|[d]|[D])MM', r'([0][0-9]|1[0-2])', x)
    x = re.sub('(?<![h]|[H]|[d]|[D])mm', r'([0][0-9]|1[0-2])', x)
    x = x.replace('SS', r'([0-5]{1}[0-9]{1})')
    x = x.replace('ss', r'([0-5]{1}[0-9]{1})')
    return x


def _helper_get_filename_ts(filename, formats):
    formats_ = dict(
        zip([_create_regex_from_pattern_2(x) for x in list(formats.values())],
            list(formats.values())
            )
    )
    formats_.update(
        dict(
            zip([_create_regex_from_pattern_2(x) for x in list(_split_timestamps(formats).values())],
                list(_split_timestamps(formats).values())
                )
        )
    )
    dd = dict(
        zip(
            formats_.values(),
            [[[(i.start(), i.end()) for i in re.finditer(key, strr)]
              for strr in [filename]]
             for key, val in formats_.items()]
        )
    )
    dd = pd.DataFrame(dd).transpose()

    dd.columns = ['starter']
    dd = dd[dd.starter.apply(lambda x: len(x) > 0)]
    dd['begin'] = dd.starter.apply(lambda x: x[0][0])
    dd['end'] = dd.starter.apply(lambda x: x[0][1])
    dd['distance'] = dd.end - dd.begin

    try:
        return dd.sort_values(['distance'], ascending=False).index[0]
    except KeyError:
        return None


def _convert_timestamp(y, formats):
    try:

        strpd_times = pd.DataFrame([[_test_ts_format(y.iloc[0], key), form_]
                                    for key, form_ in formats.items()],
                                   columns=['strpd', 'format'])

        strpd_times = strpd_times[~strpd_times.strpd.isna()]

        strpd_times['timesdiff'] = strpd_times.strpd.apply(
            lambda x: (pd.Timestamp.now() - x).total_seconds())
        strpd_times = strpd_times.sort_values('timesdiff')
        strpd_times = strpd_times[strpd_times.strpd <= pd.Timestamp.now()]
        return strpd_times['format'].iloc[0]
    except KeyError:
        return np.nan


def _get_filename_timestamps(x, formats, max_len):
    puncts = [":", "-", "_", "\\.", ",", "/", " "]
    new_forms = dict()
    for key, form in formats.items():
        for p in puncts:
            try:
                key = re.sub(p, '', key)
                form = re.sub(p, '', form)
            except TypeError:
                pass
        if len(form) <= max_len:
            new_forms[key] = form
    formats.update(_split_timestamps(formats))
    new_forms = formats
    format_ = _convert_timestamp(pd.Series(x), new_forms)
    return format_


def _split_timestamps(formats):
    split_by = [' ', "'T'"]
    new_forms = dict()
    for key, form in formats.items():
        for p in split_by:
            try:
                key_ = re.split(p, key)
                form_ = re.split(p, form)
            except TypeError:
                key_ = []
                form_ = []
                pass

            for k, v in zip(key_, form_):
                if len(key_) == 2 and len(k) > 2:
                    new_forms[k] = v
    return new_forms


def _get_regex_pattern(y):
    num_pat = re.findall('[0-9]+', y)
    if len(num_pat) > 0:
        for i in num_pat:
            y = y.replace(i, "\\d{" + str(len(i)) + "}")
    return y


def _count_iterable(i):
    return sum(1 for _ in i)


def _detect_floats_in_str(X):

    ttt = re.finditer(r"[0-9]\.[0-9]", X)
    if _count_iterable(ttt) > 0:
        X_copy = []
        patterns = []
        anti_pattern = []
        end = len(X)
        for j, i in enumerate(ttt):
            if j == 0:
                anti_pattern.append((0, i.start()))
            else:
                anti_pattern.append((end, i.start()))
            patterns.append((i.start(), i.end()))
            end = i.end()
        anti_pattern.append((end, len(X)))

        b = [(X[st:en], False) for st, en in anti_pattern]
        a = [(X[st:en], True) for st, en in patterns]
        df = X_copy.extend(list(itertools.chain.from_iterable(list(itertools.zip_longest(b, a)))))
        df.columns = ['pattern', 'is_digit']
    else:
        df = pd.DataFrame(data={'pattern': [X], 'is_digit': [False]})
    return df


def _get_major_sequences(X):

    splits = _detect_floats_in_str(X)
    splits['pattern'] = splits[~splits.is_digit].pattern.apply(lambda x: re.split(f"[{string.punctuation}]", x))
    splits = splits.explode('pattern').reset_index(drop=True)
    splits = splits[splits.pattern.astype(str).apply(lambda x: len(x) > 0)]
    splits['charcs'] = splits.pattern.str.contains('[A-Z]|[a-z]')
    splits['regex_pattern'] = splits.pattern.apply(_get_regex_pattern)
    splits['full_filename'] = X
    return splits.to_dict()


def _create_regex_from_pattern(x):
    x = x.replace('dd', '%d')
    x = re.sub(pattern=r"(?<![h]|[H]|[:])MM", repl='%m', string=x)
    x = re.sub(pattern=r"(?<![h]|[H]|[:])mm", repl='%m', string=x)
    x = x.replace('yyyy', '%Y')
    x = x.replace('yy', '%y')
    x = x.replace('HH', '%H')
    x = x.replace('hh', '%H')
    pp = string.punctuation
    x = re.sub(pattern=f'(?<![y][{pp}]{0, 1}|[Y][{pp}]{0, 1}|[d][{pp}]{0, 1}|[D][{pp}]{0, 1})mm',
               repl='%M', string=x)
    x = re.sub(pattern=f'(?<![y][{pp}]{0, 1}|[Y][{pp}]{0, 1}|[d][{pp}]{0, 1}|[D][{pp}]{0, 1})MM',
               repl='%M', string=x)
    x = x.replace('SS', '%S')
    x = x.replace('ss', '%S')
    return x


def get_possible_timestamps():
    """
    Function that defines all possible timestamps that can be discovered in the functions

    :returns dictionary of timestamp formats
    """
    possible_timestamp_formats_dict = {'%d.%m.%y %H:%M:%S': 'dd.MM.yy HH:mm:ss',
                                       "%Y-%m-%d'T'%H:%M:%S": "yyyy-MM-dd'T'HH:mm:ss",
                                       '%Y-%m-%d %H:%M:%S': 'yyyy-MM-dd HH:mm:ss',
                                       '%d/%m/%Y %H:%M:%S': 'dd/MM/yyyy HH:mm:ss',
                                       "%Y-%m-%d'T'%H:%M:%S'Z'": "yyyy-MM-dd'T'HH:mm:ss'Z'",
                                       '%d/%m/%Y %H:%M': 'dd/MM/yyyy HH:mm',
                                       "%Y-%m-%d'T'%H:%M:%SX": "yyyy-MM-dd'T'HH:mm:ssX",
                                       '%y-%m-%d %H:%M:%S': 'yy-MM-dd HH:mm:ss',
                                       '%d/%m/%y-%H:%M:%S': 'dd/MM/yy-HH:mm:ss',
                                       '%m-%d-%Y %H:%M:%S': 'MM-dd-yyyy HH:mm:ss',
                                       '%d.%m.%Y %H:%M': 'dd.MM.yyyy HH:mm',
                                       '%d.%m.%Y %H:%M:%S': 'dd.MM.yyyy HH:mm:ss',
                                       '%Y-%m-%d-%H-%m-%S': 'yyyy-MM-dd-HH-mm-ss',
                                       '%Y-%m-%d %H:%M': 'yyyy-MM-dd HH:mm',
                                       '%y%m%d %H:%M:%S': 'yyMMdd HH:mm:ss',
                                       '%Y-%m-%d %H-%m-%S': 'yyyy-MM-dd HH-mm-ss',
                                       '%Y/%m/%d %H:%M': 'yyyy/MM/dd HH:mm',
                                       '%m/%d/%Y %H:%M:%S': 'MM/dd/yyyy HH:mm:ss',
                                       '%d-%m-%Y %H:%M:%S': 'dd-MM-yyyy HH:mm:ss',
                                       '%Y-%m-%d %H:%M a Z': 'yyyy-MM-dd hh:mm a Z',
                                       "%Y%m%d'T'%H%M%S'z'": "yyyyMMdd'T'HHmmss'z'",
                                       "%Y%m%d'T'%H%M%S": "yyyyMMdd'T'HHmmss",
                                       '%m/%d/%Y %H:%M a': 'MM/dd/yyyy hh:mm a',
                                       '%d-%m-%y %H:%M:%S': 'dd-MM-yy HH:mm:ss',
                                       "%Y-%m-%d'T'%H-%M-%S": "yyyy-MM-dd'T'HH-mm-ss",
                                       '%d/%m/%y %H:%M:%S': 'dd/MM/yy HH:mm:ss',
                                       '%Y.%m.%d %H:%M:%S': 'yyyy.MM.dd HH:mm:ss',
                                       "%Y%m%d'T'%H%M%S'Z'": "yyyyMMdd'T'HHmmss'Z'",
                                       'M/%d/%Y h:%M:%S a': 'M/dd/yyyy h:mm:ss a',
                                       'M/d/%Y h:%M a': 'M/d/yyyy h:mm a',
                                       '%Y/%m/%d %H:%M:%S.S': 'yyyy/MM/dd HH:mm:ss.S',
                                       '%Y/%m/%d %H:%M:%S': 'yyyy/MM/dd HH:mm:ss',
                                       '%Y%m%d_%H%M': 'yyyyMMdd_hhmm',
                                       '%Y%m%d %H:%M': 'yyyyMMdd HH:mm',
                                       '%d/%m/%y %H:%M:%S a': 'dd/MM/yy hh:mm:ss a',
                                       '%y-%m-%d %H:%M:%S a': 'yy-MM-dd hh:mm:ss a',
                                       "%Y-%m-%d'T'%H:%M:%SXXX": "yyyy-MM-dd'T'HH:mm:ssXXX",
                                       '%d/%m/%Y%H:%M:%S': 'dd/MM/yyyyHH:mm:ss',
                                       '%Y%m%d%H%M%S': 'yyyyMMddHHmmss',
                                       '%Y-%m-%d': 'yyyy-MM-dd',
                                       '%d/%m/%Y': 'dd/MM/yyyy',
                                       '%H:%M %d/%m/%y': 'HH:mm dd/MM/yy',
                                       '%Y%m%d%H%M': 'yyyyMMddHHmm',
                                       "%Y-%m-%d'T'%H:%M:%SZ": "yyyy-MM-dd'T'HH:mm:ssZ",
                                       '%d-%m-%Y %H:%M': 'dd-MM-yyyy HH:mm',
                                       "%d.%m.%y '/' %H:%M:%S": "dd.MM.yy '/' HH:mm:ss",
                                       '%Y%m%d %H:%M:%S': 'yyyyMMdd HH:mm:ss',
                                       '%d/%m/%Y %H%M': 'dd/MM/yyyy HHmm',
                                       "%d/%m/%Y'T'%H:%M": "dd/MM/yyyy'T'HH:mm",
                                       '%d %mM %y': 'dd MMM yy',
                                       '%Y%md-%H%M%S': 'yyyyMMd-HHmmss',
                                       '%H%M%S': 'hhmmss',
                                       '%Y-%m-%d_%H:%M:%S': 'yyyy-MM-dd_HH:mm:ss',
                                       '%Y%m%d_%H%M%S': 'yyyyMMdd_HHmmss',
                                       '%Y%md_%H%M': 'yyyyMMd_HHmm',
                                       '%d-%m-%Y': 'dd-MM-yyyy',
                                       "%y%m%d'T'%H%M%S": "yyMMdd'T'HHmmss",
                                       '%Y%m%d': 'yyyyMMdd',
                                       '%Y%m%d-%H%M': 'yyyyMMdd-HHmm',
                                       '%Y%m%d_%H_%m': 'yyyyMMdd_HH_mm',
                                       '%Y%m%dT%H%M%S': 'yyyyMMddTHHmmss',
                                       '%Y-%m-%d_%H%M%S': 'yyyy-MM-dd_HHmmss',
                                       '%Y_%m_%d_%H_%m': 'yyyy_MM_dd_HH_mm',
                                       '%y%m%d': 'yyMMdd',
                                       '%d_%m_%Y': 'dd_MM_yyyy',
                                       '%Y%m%d-%H%M%S': 'yyyyMMdd-HHmmss',
                                       '%Y%md_%H%M%S': 'yyyyMMd_HHmmss',
                                       '%Y%m%d-%H:%M:%S': 'yyyyMMdd-HH:mm:ss',
                                       '%y-%m-%d %H_%m': 'yy-MM-dd HH_mm',
                                       '%Y-%m-%d_%H-%m-%S': 'yyyy-MM-dd_HH-mm-ss',
                                       '%Y%md_%H%Md': 'yyyyMMd_HHmmd',
                                       "%Y%m%S'T'%H%M%S": "yyyyMMss'T'HHmmss",
                                       '%Y.%m.%d': 'yyyy.MM.dd',
                                       '%d.%m.%Y': 'dd.MM.yyyy',
                                       '%d%m%y': 'ddMMyy',
                                       '%Y%m': 'yyyyMM',
                                       '%Y_%m_%d': 'yyyy_MM_dd'}
    return possible_timestamp_formats_dict


def _get_filename_splits(df, column, initial_split=False):
    possible_timestamp_formats_dict = get_possible_timestamps()
    majors = df[column].apply(_get_major_sequences)
    majors = pd.concat([pd.DataFrame.from_dict(majors.iloc[i]) for i in range(majors.shape[0])])
    majors['file'] = (majors.reset_index()['index'].shift(1) > majors.reset_index()['index']).tolist()
    majors['file'] = majors.file.cumsum()
    majors.index.name = 'file_position'
    majors['dupli'] = majors.groupby(level='file_position').nunique()['pattern'].rename_axis('dupli')

    if not initial_split:
        max_len = majors[majors.dupli > 1].pattern.apply(lambda x: len(x)).max()
        formats = possible_timestamp_formats_dict
        formats.update(_split_timestamps(formats))
        majors['timestamp_format'] = majors.pattern.apply(
            lambda x: _get_filename_timestamps(x, formats, max_len=max_len))
        if majors.dropna().groupby('file').timestamp_format.count().max() > 1:
            date_ = [(x, datetime.strptime(majors.pattern.loc[x].iloc[0],
                                           _create_regex_from_pattern(majors.timestamp_format.loc[x].iloc[0])))
                     for x in majors[majors.file == 0].dropna().index]
            date_ = dict(date_)
            for key, value in date_.items():
                try:
                    date_[key] = abs((pd.Timestamp.now() - value).total_seconds())
                except KeyError:
                    date_[key] = np.inf
            date_ = {key: value for key, value in sorted(date_.items(), key=lambda item: item[1])}
            date_.pop(list(date_.keys())[0])
            for k in date_.keys():
                majors.loc[k, 'timestamp_format'] = np.nan

    return majors


def _get_punctuations(filename, majs, groupby_id):
    k = filename[groupby_id].tolist()
    patterns = majs[majs.full_filename == k[0]].pattern.tolist()
    for x in patterns:
        k = list(itertools.chain.from_iterable([j.split(x, 1) for j in k]))
    if len(k[0]) == 0:
        k = k[1:]
    return k


def _build_name(x, k):
    Y = x.copy()
    if not isinstance(k, list):
        k = k[Y['full_filename'].unique()[0]]

    Y['regex'] = Y.apply(lambda z: z.pattern if pd.isna(z.regex_pattern) else z.regex_pattern, axis=1)
    if len(k) < Y.shape[0]:
        k = [j for j in k if len(j) > 0]
        k.append('')

    k = ["" + j if len(j) > 0 else "" for j in k]
    ret_str = ''.join(list(itertools.chain.from_iterable(zip(Y['regex'].tolist(), k))))
    return ret_str


def merge_filename_parts(df, majors, groupby_c='file', groupby_id='filename'):
    """
    Use the single parts that the filenames were split into to build a regular expression

    :param df: DataFrame with filenames
    :param majors: DataFrame with major parts of the filenames
    :param groupby_c: The grouping identifier that should not be changed
    :param groupby_id: The grouping id for the particular dataframe
    :return:
    """
    k = df.groupby(groupby_id).apply(lambda x: _get_punctuations(x, majors, groupby_id))

    regex = majors.groupby(groupby_c).apply(lambda x: _build_name(x, k=k))
    if len(regex.unique()) == 1:
        regex = regex[0]
        print(f'regexpr identifier is: {regex}')
    else:
        regex = regex

    df['regex_identifier'] = regex
    try:
        df['filename_date'] = majors.timestamp_format.dropna().unique()[0]
    except AttributeError:
        pass
    return df


def _add_timestamp_format(filenames_):
    possible_timestamp_formats_dict = get_possible_timestamps()
    ret = pd.Series(filenames_).apply(
        lambda x: _helper_get_filename_ts(x, possible_timestamp_formats_dict))
    ret = ret[ret.apply(lambda x: x is not None and x != [])].to_frame(name='ts_format')
    ret['occurrence'] = 1
    ret['len'] = ret.ts_format.apply(len)

    ret = (ret
           .groupby('ts_format')
           .agg({'occurrence': 'count', 'len': 'mean'})
           .sort_values(['occurrence', 'len'], ascending=False)
           .index[0])
    return ret


def _find_max(x):
    xx = x.apply(pd.Series)
    xx = xx.applymap(lambda z: z.replace(r'd{', ''))
    xx = xx.applymap(lambda z: int(z.replace(r'}', '')))
    xx = xx.apply(lambda z: f'{z.min()}, {z.max()}' if z.min() != z.max() else f'{z.min()}', axis=0)
    xx = [r'\d{' + y + '}' for y in xx]
    return xx


def _merge_integers(extracted):
    nums = [re.findall(pattern=r'd\{[1-9]+\}', string=x) for x in extracted.regex_identifier]
    parts = [r"***___***".join(re.split(pattern=r'd\{[1-9]+\}', string=x)) for x in extracted.regex_identifier]
    replced = (pd.DataFrame(data={'nums': nums, 'parts': parts})
               .groupby(parts)
               .nums
               .apply(_find_max)
               .to_frame('new_int_reg')
               )
    org_ = (pd.DataFrame(data={'regex_identifier': extracted.regex_identifier, 'parts': parts})
            .set_index('parts')
            .join(replced)
            .reset_index()
            )

    org_['repl'] = org_.apply(lambda x: re.sub(re.escape(r'\***___***'), r'{}', x['index']).format(*x['new_int_reg']),
                              axis=1)
    extracted = (extracted
                 .merge(org_[['regex_identifier', 'repl']])
                 .drop(columns='regex_identifier')
                 .rename(columns={'repl': 'regex_identifier'})
                 )
    return extracted


def regex_pattern_extraction(lines, column='filename'):
    """
    Function to extract common parts of strings and replace the varying parts with regular expressions

    :param lines: DataFrame that contains the single lines of text
    :param column: The column name to analyze. Default: 'filename'
    :return:
    """
    majors = _get_filename_splits(df=lines, column=column, initial_split=True)
    filenames_ = merge_filename_parts(lines, majors, groupby_id=column)
    filenames_ = _merge_integers(filenames_)
    filenames_ = filenames_.groupby('regex_identifier')[column].apply(lambda x: x.tolist()).reset_index()
    filenames_['timestamp_format'] = filenames_[column].apply(_add_timestamp_format)
    filenames_ = filenames_.sort_values('regex_identifier')

    return filenames_
