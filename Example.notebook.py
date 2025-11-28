import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import hmm_tagger
    return (hmm_tagger,)


@app.cell
def _(hmm_tagger):
    hmm_tagger
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ejemplo
    """)
    return


@app.cell
def _():
    print("hello")
    return


if __name__ == "__main__":
    app.run()
