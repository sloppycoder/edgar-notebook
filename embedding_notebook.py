import marimo

__generated_with = "0.10.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import polars as pl
    import marimo as mo
    from sleuth.datastore import execute_query

    def run_query(sql):
        result = execute_query(sql)
        print(result)
        return pl.DataFrame(result)

    mo.md("""
    ### initialization, define common functions
    """)
    return execute_query, mo, pl, run_query


@app.cell(hide_code=True)
def _(pl):
    from sleuth.llm.algo import (
        most_relevant_chunks,
        relevance_by_appearance,
        relevance_by_distance,
    )

    from sleuth.trustee import get_relevant_chunks_with_distances

    tag = "123"

    def run_vector_search(cik, accession_number):
        result = get_relevant_chunks_with_distances(
            cik=cik, accession_number=accession_number,    
            embedding_table_name="filing_chunks_embeddings",
            search_phrase_table_name ="search_phrase_embeddings",
            search_phrase_tag="gemini_768",
            embedding_tag=tag,
        )
        return pl.DataFrame(result)

    run_vector_search("1002427","0001133228-24-004879")
    return (
        get_relevant_chunks_with_distances,
        most_relevant_chunks,
        relevance_by_appearance,
        relevance_by_distance,
        run_vector_search,
        tag,
    )


@app.cell(hide_code=True)
def _(mo):
    filings  = {"1002427 / 0001133228-24-004879" : ["199","200"]}


    filing_dropdown = mo.ui.dropdown(options=filings.keys(), label="select a filing")

    mo.md(f"""
    #### change the filtered_df above to examine different scenarios
    {filing_dropdown}
    """)
    return filing_dropdown, filings


@app.cell(hide_code=True)
def _(filing_dropdown, filings, mo):
    chunks = filings[filing_dropdown.value] if filing_dropdown.value else []
    chunk_dropdown = mo.ui.dropdown(options=chunks, label="select chunk to view")
    chunk_dropdown
    return chunk_dropdown, chunks


@app.cell
def _(chunk_dropdown, filing_dropdown, mo, tag):
    from sleuth.trustee import get_text_by_chunk_num
    selected_text=""
    if chunk_dropdown.value:
        _tmp1 = filing_dropdown.value.split("/")
        cik, accession_number  = _tmp1[0].strip(), _tmp1[1].strip()
        chunk_nums = [int(n) for n in chunk_dropdown.value.split(",")]
        selected_text = get_text_by_chunk_num(text_table_name="filing_text_chunks",chunk_nums=chunk_nums,
                                              tag=tag,cik=cik,accession_number=accession_number)

    mo.ui.text_area(selected_text, rows=20)
    return (
        accession_number,
        chunk_nums,
        cik,
        get_text_by_chunk_num,
        selected_text,
    )


@app.cell
def _(filing_dropdown, mo):
    _tmp2 = filing_dropdown.value.split("/")
    cik_input = mo.ui.text(_tmp2[0].strip())
    accession_number_input = mo.ui.text(_tmp2[1].strip())
    mo.md(f"""
    ### Select filing to examine
    {cik_input} {accession_number_input}
    """)
    return accession_number_input, cik_input


@app.cell
def _():
    # search_df = run_vector_search(cik_input.value, accession_number_input.value)
    # chunk_distances = gather_chunk_distances(search_df)
    # print(relevance_by_distance(chunk_distances))
    # print(relevance_by_appearance(chunk_distances))
    # search_df
    return


if __name__ == "__main__":
    app.run()
