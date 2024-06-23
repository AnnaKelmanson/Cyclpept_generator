from dataframe_handler import final_sequence
import pandas as pd
import warnings
import sys
import os

from models.apply_model import make_predictions
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdCoordGen

from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
path_to_cyclpept_ml = os.path.abspath('cyclpept-ML-models')
sys.path.append(path_to_cyclpept_ml)


def initialize_reaction(sequence):
    sequence = sequence.split(' ')
    sequence.reverse()
    return sequence, AllChem.ReactionFromSmarts('([#8]=[#6](-[#7:1])-[#8]-[#6]-[#6]1-[#6]2:[#6](-[#6]3:[#6]-1:[#6]:[#6]:[#6]:[#6]:3):[#6]:[#6]:[#6]:[#6]:2).([#8:3]=[#6:2]-[OH])>>[#8:3]=[#6:2]-[#7:1]')

def perform_initial_coupling(sequence, bbdict, rxn):
    reacts = (Chem.MolFromSmiles(bbdict[sequence[0]]), Chem.MolFromSmiles(bbdict[sequence[1]]))
    products = rxn.RunReactants(reacts)
    return products[0][0]  # Assuming the reaction always succeeds and takes the first product

def extend_peptide_chain(sequence, product, bbdict, rxn):
    for aa in sequence[2:]:
        try:
            reactant_smiles = bbdict[aa]
            reactant_mol = Chem.MolFromSmiles(reactant_smiles)
        except KeyError:
            print(f"KeyError: '{aa}' not found in dictionary.")
            break  # Adjust based on desired error handling

        reacts = (product, reactant_mol)
        products = rxn.RunReactants(reacts)
        if products:
            product = products[0][0]  # Assuming successful reaction
        else:
            #print(f"Error: There is an issue with {aa}")
            try:
                rxn2 = AllChem.ReactionFromSmarts('([#8]-[#6](=[#7:1])-[#8]-[#6]-[#6]1-[#6]2:[#6](-[#6]3:[#6]-1:[#6]:[#6]:[#6]:[#6]:3):[#6]:[#6]:[#6]:[#6]:2).([#8:3]=[#6:2]-[OH])>>[#8:3]=[#6:2]-[#7:1]')
                products = rxn2.RunReactants(reacts)
                product = products[0][0]
            except:
                print(f"Error: There is an issue with {aa}")
                break  # Adjust based on desired error handling
    return product

def remove_protecting_groups(product):
    # Remove nboc
    nboc_smarts = 'CC(C)(C)OC(NC)=O'
    if product.HasSubstructMatch(Chem.MolFromSmiles(nboc_smarts)):
        rxn_nboc = AllChem.ReactionFromSmarts('([#6]-[#6](-[#6])(-[#6])-[#8]-[#6](-[#7:1]-[#6:2])=[#8])>>[#7:1]-[#6:2]')
        product = rxn_nboc.RunReactants((product,))[0][0]

    # Replace substructures for tbu
    tbu_smarts = 'COC(C)(C)C'
    tbu_replacement = 'CO'
    product = Chem.rdmolops.ReplaceSubstructs(product, Chem.MolFromSmiles(tbu_smarts), Chem.MolFromSmiles(tbu_replacement))[0]

    # Delete substructures for pbf and trt
    for pg_smarts in ['CC1(C)Cc2c(C)c(S(=O)=O)c(C)c(C)c2O1', 'C(c1ccccc1)(c2ccccc2)c3ccccc3']:
        product = Chem.rdmolops.DeleteSubstructs(product, Chem.MolFromSmiles(pg_smarts))

    # FMOC removal (if necessary)
    fmoc_smarts = 'O=COCC1C2=C(C=CC=C2)C3=C1C=CC=C3'
    product = Chem.rdmolops.DeleteSubstructs(product, Chem.MolFromSmiles(fmoc_smarts))

    return product

def build_peptide(sequence, bbdict):
    sequence, rxn = initialize_reaction(sequence)
    product = perform_initial_coupling(sequence, bbdict, rxn)
    if len(sequence) > 2:
        product = extend_peptide_chain(sequence, product, bbdict, rxn)
    product = remove_protecting_groups(product)
    
    return product
#Chem.MolToSmiles(product, kekuleSmiles=True)

def find_largest_cycle(products):

    mols = []
    for product in products:
        mols.append(product[0])
    largest_cycle_size = 0
    mol_with_largest_cycle = None
    
    for mol in mols:
        sssr = rdmolops.GetSSSR(mol)  # Get the smallest set of smallest rings
        max_ring_size = max((len(ring) for ring in sssr), default=0)
        
        if max_ring_size > largest_cycle_size:
            largest_cycle_size = max_ring_size
            mol_with_largest_cycle = mol
    
    return mol_with_largest_cycle

def disulfide_cyc(linear_peptide):
    
    oxrxn = AllChem.ReactionFromSmarts('([#16:1].[#16:2])>>[#16:1]-[#16:2]')
    reacts = (linear_peptide,)
    try:
        products = oxrxn.RunReactants(reacts)

        product = find_largest_cycle(products) 
        
        rdCoordGen.AddCoords(product) #makes mc look nicer
    except:
        print('Vot nezadacha')

    return product

def linker_cyc(linear_peptide, linker):
    
    cycrxn = AllChem.ReactionFromSmarts('([#16:1]-[#6:2].[#16:3]-[#6:4]).([#6:5]-[#17].[#6:6]-[#17])>>([#6:2]-[#16:1]-[#6:5].[#6:4]-[#16:3]-[#6:6])')
    reacts = (linear_peptide, linker)
    try:
        products = cycrxn.RunReactants(reacts) 
        product = find_largest_cycle(products)  
        rdCoordGen.AddCoords(product)
    except:
        print('Quel dommage!')

    return product


def build_peptide_from_list(aa_list, bbdict, type_of_cyclization):
    """
    Build a peptide from a list of amino acids using RDKit, handling cyclization and protection group removal.

    Parameters:
    - aa_list (list): List of amino acid identifiers.
    - bbdict (dict): Dictionary mapping amino acid identifiers to SMILES strings.

    Returns:
    - RDKit Mol: The final molecular structure after all transformations.
    """
    linker = Chem.MolFromSmiles(bbdict['linker1'])
    sequence = ' '.join(aa_list)  # Create a sequence string from list
    sequence = final_sequence(sequence)
    linear_peptide = build_peptide(sequence, bbdict)
    if type_of_cyclization == 'disulfide':
        product = disulfide_cyc(linear_peptide)
    else:
        product = linker_cyc(linear_peptide, linker)
    return product

def create_df_row_for_child_structure(seq, bbdict, type_of_cyclization):
    product = build_peptide_from_list(seq, bbdict, type_of_cyclization)
    product.UpdatePropertyCache()
    #product = Chem.AddHs(product)
    Chem.SanitizeMol(product) #,Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    smiles = Chem.MolToSmiles(product)
    permeability = make_predictions([smiles])
    permeability = permeability[0]

    df = pd.DataFrame({
        'Sequence': [seq],  
        'Mol': [product],
        'SMILES': [smiles],
        'Permeability': [permeability]
    })

    return df