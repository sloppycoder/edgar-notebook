import marimo

__generated_with = "0.10.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import polars as pl
    import marimo as mo
    from sleuth.datastore import execute_query
    import json
    from dotenv import load_dotenv

    load_dotenv()

    def run_query(sql):
        result = execute_query(sql)
        print(result)
        return pl.DataFrame(result)

    mo.md("""
    ### initialization, define common functions
    """)
    return execute_query, json, load_dotenv, mo, pl, run_query


@app.cell(hide_code=True)
def _(execute_query, pl):
    from sleuth.llm.algo import (
        most_relevant_chunks,
        relevance_by_appearance,
        relevance_by_distance,
        gather_chunk_distances,
    )

    from sleuth.trustee import relevant_chunks_with_distances

    tag = "orig225"
    search_tag = "orig225"

    all_filings = []

    _result = execute_query(
        "select distinct cik, accession_number from filing_chunks_embeddings order by 1,2"
    )
    for _row in _result:
        _relevance_result = relevant_chunks_with_distances(
            cik=_row["cik"],
            accession_number=_row["accession_number"],
            embedding_table_name="filing_chunks_embeddings",
            search_phrase_table_name="search_phrase_embeddings",
            search_phrase_tag=search_tag,
            embedding_tag=tag,
        )
        _chunk_distances = gather_chunk_distances(_relevance_result)
        _by_appearance = most_relevant_chunks(relevance_by_appearance(_chunk_distances))
        _by_distance = most_relevant_chunks(relevance_by_distance(_chunk_distances))

        all_filings.append(
            {
                "cik": _row["cik"],
                "accession_number": _row["accession_number"],
                "relevance_result": _relevance_result,
                "by_appearance": _by_appearance,
                "by_distance": _by_distance,
            }
        )

    print(f"Loaded {len(all_filings)} filings")

    filings = pl.DataFrame(all_filings)
    filings = filings.with_columns(
        pl.concat_str(
            [pl.col("cik"), pl.col("accession_number")], separator=" / "
        ).alias("key")
    )
    filings
    return (
        all_filings,
        filings,
        gather_chunk_distances,
        most_relevant_chunks,
        relevance_by_appearance,
        relevance_by_distance,
        relevant_chunks_with_distances,
        search_tag,
        tag,
    )


@app.cell
def _(filings, mo, pl):
    filtered_filings = filings.filter(
        (pl.col("by_distance") != pl.col("by_appearance"))
    )
    # filtered_filings = filings
    mo.ui.dataframe(filtered_filings)
    return (filtered_filings,)


@app.cell(hide_code=True)
def _(filtered_filings, mo):
    filing_keys = filtered_filings["key"].to_list()

    filing_dropdown = mo.ui.dropdown(options=filing_keys, label="select a filing")

    mo.md(f"""
    #### change the filtered_df above to examine different scenarios
    {filing_dropdown}
    """)
    return filing_dropdown, filing_keys


@app.cell(hide_code=True)
def _(filing_dropdown, filtered_filings, mo):
    _chunks = []
    if filing_dropdown.value:
        _tmp3 = filtered_filings.filter(
            filtered_filings["key"] == filing_dropdown.value
        ).select(["cik", "accession_number", "by_distance", "by_appearance"])
        selected_row = dict(zip(_tmp3.columns, _tmp3.row(0)))
        # print(selected_row)
        _chunks = [str(selected_row["by_distance"]), str(selected_row["by_appearance"])]

    chunk_text_input = mo.ui.text(placeholder="enter chunk number or range")
    chunk_dropdown = mo.ui.dropdown(
        options=_chunks,
        label="select chunk to view, first by distance, second by appearance",
    )
    mo.md(f"""
    {chunk_dropdown}
    {chunk_text_input}
    """)
    return chunk_dropdown, chunk_text_input, selected_row


@app.cell(hide_code=True)
def _(
    chunk_dropdown,
    chunk_text_input,
    execute_query,
    json,
    mo,
    selected_row,
    tag,
):
    _selected_text = ""
    if chunk_dropdown.value:
        if chunk_text_input.value:
            _chunk_nums = [int(s) for s in chunk_text_input.value.split(",")]
        else:
            _chunk_nums = json.loads(chunk_dropdown.value)

        _ret = execute_query(
            f"""
        SELECT
            STRING_AGG('✳️✳️✳️✳️ ' || chunk_num || ' ✳️✳️✳️✳️\n' || chunk_text, '\n' ORDER BY chunk_num) as relevant_text
        FROM filing_text_chunks
            WHERE cik = %s AND accession_number = %s AND %s = ANY(tags) AND chunk_num = ANY(%s)
        """,
            (selected_row["cik"], selected_row["accession_number"], tag, _chunk_nums),
        )
        _selected_text = _ret[0]["relevant_text"]

    mo.ui.text_area(_selected_text, rows=20)
    return


@app.cell(hide_code=True)
def _(mo, selected_row):
    cik_input = mo.ui.text(selected_row["cik"])
    accession_number_input = mo.ui.text(selected_row["accession_number"])
    mo.md(f"""
    ### Select filing to examine
    {cik_input} {accession_number_input}
    """)
    return accession_number_input, cik_input


@app.cell(hide_code=True)
def _(pl, relevant_chunks_with_distances, search_tag, selected_row, tag):
    pl.DataFrame(
        relevant_chunks_with_distances(
            cik=selected_row["cik"],
            accession_number=selected_row["accession_number"],
            embedding_table_name="filing_chunks_embeddings",
            search_phrase_table_name="search_phrase_embeddings",
            search_phrase_tag=search_tag,
            embedding_tag=tag,
        )
    )
    return


if __name__ == "__main__":
    app.run()
