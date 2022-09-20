# ©, 2021, Sirris
# owner: KVVR

"""table visualizations using plotly"""
import dash_table
from jupyter_dash import JupyterDash
import dash_html_components as html


def plot_table(df,
               columns=None,
               columns_deletable=False,
               columns_renamable=False,
               columns_hidable=False,
               row_deletable=False,
               columns_sort=True,
               data_editable=False,
               enable_data_export=False,
               enable_filtering=True,
               show_all_data=False,
               page_size=250,
               round_numbers=None,
               port=8050):
    """
    Pretty plot dataframe content in a interactive table using JupyterDash

    (Note: pip install dash, jupyterdash)
    :param df: pandas.DataFrame. A dataframe containing data to plot in a table
    :param columns: None or list (default: None).
        A list of the columns to print from the given dataframe df
    :param columns_deletable: boolean (default: False).
        If True, columns can be deleted from the view
        (Note: they will not be used when exporting if export is enabled)
    :param columns_renamable: boolean (default: False).
        If True, column names can be renamed
    :param columns_hidable: boolean (default: False).
        If True, columns can be hidden from the table view
        (Note: These columns will still be used when exporting if export is
        enabled)
    :param row_deletable: boolean (default: False).
        If True, a row of data can be removed from the table view
    :param columns_sort: boolean (default: True)
        If True, the columns can be sorted
    :param data_editable: boolean (default: False).
        If True, the data in the table view is editable
    :param enable_data_export: boolean (default: False)
        If True, an export button will be added to export the data
        (Note: deleted rows, columns will not be exported, hidden columns will
        be)
    :param enable_filtering: boolean (default: True)
        If True, data can be filtered per column
    :param show_all_data: boolean (default: False).
        If True, the whole dataset will be printed in one table
        (Note: can take a long time with large dataframe)
    :param page_size: int (default: 250)
        The number of rows per page
        (Note: if show_all_data is set to True, this is ignored)
    :param round_numbers: int (default: None).
        The number of decimals for the floats. Default will not round floats
    :param port: int (default:8050).
        The port used to render the dash app
        (Note: alter when port is already used)
    :return: none
    """
    app = JupyterDash(__name__)
    if columns:
        df = df[columns]
    if round_numbers:
        df = df.round(round_numbers)
    app.layout = html.Div([
        dash_table.DataTable(
            style_cell={
                'whiteSpace': 'normal',
                'height': 'auto'},
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto'},
            id='datatable-interactivity',
            columns=[
                {'name': i, 'id': i,
                 'deletable': columns_deletable, 'renamable': columns_renamable,
                 'hideable': columns_hidable}
                for i in df.columns],
            sort_action=('native' if columns_sort else 'none'),
            data=df.to_dict('records'),
            editable=data_editable,
            row_deletable=row_deletable,
            filter_action=('native' if enable_filtering else 'none'),
            page_action='none' if show_all_data else 'native',
            page_size=page_size,
            export_format=('csv' if enable_data_export else 'none'),
        )],
    )
    app.run_server(mode='inline', port=port)
