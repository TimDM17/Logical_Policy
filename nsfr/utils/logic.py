from nsfr.fol.data_utils import DataUtils

def get_lang(lark_path, lang_base_path, dataset):
    """
    Load the language of first-order logic from files.

    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language
    """
    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path, dataset=dataset)
    lang = du.load_language()
