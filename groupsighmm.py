import numpy as np
from pomegranate import *
import re
import os
import pickle
import matplotlib.pyplot as plt
import networkx
import sklearn.metrics as sk_met
import math

def fasta_file_to_dict(filename_data):
    '''
    1. Reads fasta file
    2. Creates a dictionary of the form
    {'uniprot AC1':{'organism': ""NEGATIVE"" , 'type': ""SP"" , 'sequence': ""MNKI...""
                                                  , 'labels': ""SSSSSS....""},
     'uniprot AC2':{'organism': ""EUKARYA"" , 'type': ""NO_SP"" , 'sequence': ... , 'labels': ...},
                                                      ... }
    enabling lookup of organism, protein type, sequence and cleavage site labels by uniprot AC. Returns the total dictionary
    '''
    # seq_data = input('Enter seq. file:')
    fh = open(filename_data, 'r')
    text = fh.read()
    fastas = filter(None,text.split(">")) #Splits into list of fasta representations of individual proteins

    fasta_dict = {}
    for fasta in fastas:
        identity,sequence,labels = filter(None,fasta.split('\n'))
        uniprot_ac,organism,type,part_no = identity.split('|')
        fasta_dict[uniprot_ac] = {}
        fasta_dict[uniprot_ac]['labels'] = labels
        fasta_dict[uniprot_ac]['organism'] = organism
        fasta_dict[uniprot_ac]['type'] = type
        fasta_dict[uniprot_ac]['sequence'] = sequence

    return fasta_dict

def amino_acid_seq_to_group_seq(aa_sequence):
    ''''
    Converts amino acid sequences to their equivalent amino acid group sequences.
    '''
    group_a = ['D', 'E', 'H', 'K', 'N', 'Q', 'R']
    group_b = ['G', 'P', 'S', 'T', 'Y']
    group_c = ['F', 'I', 'L', 'M', 'V', 'W']
    group_d = ['A', 'C']

    for aa in group_a:
        aa_sequence = aa_sequence.replace(aa,'a')
    for aa in group_b:
        aa_sequence = aa_sequence.replace(aa,'b')
    for aa in group_c:
        aa_sequence = aa_sequence.replace(aa,'c')
    for aa in group_d:
        aa_sequence = aa_sequence.replace(aa,'d')

    return aa_sequence

def manual_assignment_of_states_sp(grp_sequence,labels, ignore = False):
    '''
    Rudimentary assignment of h-,n- and c-regions in signal peptide sequence. To be used for initial distribution
    of amino acid groups in the three regions and transition probabilities. These are then used to initialize the HMM.
    Follows this procedure:
    (1) Place a pointer at the -1 position
    (immediately before the cleavage site), set the assignment to
    c-region, and scan the sequence upstream towards the Nterminus;
    (2) move the pointer three positions upstream (assigning
    -1 through -3 as a minimal c-region); (3) set the
    assignment to h-region at the first occurrence of at least two
    consecutive group 'c' amino acids; (4) move the pointer six positions upstream;
    (5) set the assignment to n-region at the first occurrence of
    either a group 'a' residue or at least two consecutive group'b'
    residues; (6) if the N-terminal end of the hregion
    is not a hydrophobic residue, move the pointer back
    downstream, changing the assignment to n-region until a hydrophobic
    residue is found.
    '''
    cleavage_site = re.search('[OI]',labels).start()
    signal_pep_seq = grp_sequence[:cleavage_site]
    reversed_signal_pep_seq = signal_pep_seq[::-1] #reverses the signal peptide grp_sequence
                                                        # because we scan from the cleavage stie towards N-terminus
    regions = []
    states = []

    hydrophob_and_helix = '[c]'
    charged_or_helix_breaking = '[a]'
    generally_helix_breaking = '[b]'

    min_c_length = 3
    min_h_length = 6
    max_h_length = 20

    first_hregion = re.search(hydrophob_and_helix + '{2}',reversed_signal_pep_seq[min_c_length:len(reversed_signal_pep_seq)-7])

    if not first_hregion:
        first_hregion = re.search(hydrophob_and_helix + '{1}',reversed_signal_pep_seq[min_c_length:len(reversed_signal_pep_seq)-7])
    if not first_hregion:
        return ignore #Does not try to assign regions if a h-region simply can not be found, skips this sample. Very few such cases

    first_h = first_hregion.start()

    for i in range(0,first_h + min_c_length):
        regions.append('c')

    first_nregion = re.search(charged_or_helix_breaking + '|(' + generally_helix_breaking+ '{2})', reversed_signal_pep_seq[first_h + min_h_length + min_c_length:])

    if first_nregion and first_nregion.start() <= max_h_length - min_h_length: #sets maximum length for h-region
        first_n = first_nregion.start()
    elif first_nregion and first_nregion.start() > max_h_length - min_h_length:
        first_n = max_h_length - min_h_length
        print('unusually long h-region in: ',reversed_signal_pep_seq)
    else:
        if first_h + min_c_length + max_h_length > len(reversed_signal_pep_seq) - 1:
            first_n = len(reversed_signal_pep_seq) - 1 - first_h - min_c_length - min_h_length
        else:
            first_n = max_h_length - min_h_length
            print('unusually long h-region in: ', reversed_signal_pep_seq)



    if re.match(hydrophob_and_helix,reversed_signal_pep_seq[first_h + first_n + min_c_length + min_h_length - 1]):
        for i in range(0,first_n + min_h_length):
            regions.append('h')
        for i in range(first_h + first_n + min_c_length + min_h_length,len(reversed_signal_pep_seq)):
            regions.append('n')
    else:
        n = 0
        while n < first_n\
                and not re.match(hydrophob_and_helix,reversed_signal_pep_seq[first_h + first_n +
                                                                  min_c_length + min_h_length - 1 - n]):
            n = n + 1
        for i in range(0,first_n + min_h_length - n):
            regions.append('h')
        for i in range(first_h + first_n + min_c_length + min_h_length - n,len(reversed_signal_pep_seq)):
            regions.append('n')

    amount_h_states = 20
    highest_n_state = 8
    highest_c_state = 10
    '''
    Converts the assigned regions to individual states (c1,c2,..,n6,n5,...etc) according to the HMM model architecture, 
    allows initialization of transition parameters in addition to initialization of emission distribution.
    '''
    state_num = 0
    c_num = 1
    h_num = amount_h_states
    n_num = highest_n_state

    while state_num < len(regions):
        if regions[state_num] == 'c':
            states.append('c' + str(c_num))
            if c_num < highest_c_state:
                c_num = c_num + 1
        elif regions[state_num] == 'h' and regions[state_num + 1] == 'h':
            states.append('h' + str(h_num))
            h_num = h_num - 1
            if h_num <= 0:
                print('Odd occurence with h_num:', h_num, 'for: ', grp_sequence, 'with labels: ',labels)
        elif regions[state_num] == 'h' and regions[state_num + 1] != 'h':
            states.append('h1')
        elif regions[state_num] == 'n':
            if state_num + 1 == len(regions):
                states.append('n1')
            else:
                if n_num == 2 and len(regions) - state_num > 2:
                    states.append('n' + str(n_num))
                else:
                    states.append('n' + str(n_num))
                    n_num = n_num - 1
        state_num = state_num + 1

    states = states[::-1] #reverses the assigned states back to original orientation
    for i in range(1,5): #adding the four states after the cleavage site, where the emission distribution is unlike
                        # background distribution for rest of non-sp protein
        states.append('m' + str(i))
    for i in range(len(reversed_signal_pep_seq) + 4, len(grp_sequence)):
        states.append('m5')
    return states

def manual_assignment_of_states_anch(labels, ignore=False):
    '''
    Simpler assignment for signal anchor states.
    Assign all transmembrane aa's as h-region, all the ones before as n-region.
    A first in each state for "Anchor"
    '''

    trans_mem_region = re.search('M+', labels)
    states = []
    n_num = 6
    h_num = 12
    if trans_mem_region.end()-trans_mem_region.start() < h_num:
        return ignore
    for i in range(trans_mem_region.start(),trans_mem_region.end()):
        states.append('Ah' + str(h_num))
        if h_num > 1:
            h_num -= 1
    states[-1] = 'Ah1'
    for i in range(0,trans_mem_region.start()):
        states.append('An' + str(n_num))
        if n_num > 2:
            n_num -= 1
    states[-1] = 'An1'
    states = states[::-1]
    return states


def deduce_initial_emission_and_trans_dist(grp_sequences,state_labels,type):
    '''
    Takes in sequences of amino acid groups for proteins with signal peptides (or signal anchors) as well as their assigned
    state labels (after manual assignment) and calculates emission and transition probabilities to be used as
    initialization for the nodes and edges of the hmm.
    Returns probability of emitting a given amino acid group in a given state, in the dictionary form
    {'state1': {'amino acid 1': x1, amino acid 2': x2,....},'state2':{'amino acid 1': y1,....},...}
    where x and y are probabilities normalized for the state.
    e.g.
    {'c1': {'a': 0.02, 'b':0.32,.....}, 'h1': {'a':......},.....}
    Returns probability of transition from one state with several possible transitions to one of those states, in a
    similar dictionary form
    {'state1:{'state2':0.2,'state3':0.04,...},....}
    '''
    aa_groups = {'a':0,'b':0,'c':0,'d':0}
    emission_prob_sp = {'c': aa_groups.copy(), 'h': aa_groups.copy(), 'n': aa_groups.copy(),'c1':aa_groups.copy(),
                     'c2':aa_groups.copy(), 'c3':aa_groups.copy(),'n1':aa_groups.copy(), 'm1':aa_groups.copy(),
                     'm2':aa_groups.copy(),'c4':aa_groups.copy(), 'c5':aa_groups.copy(), 'c6':aa_groups.copy(),
                     'm3':aa_groups.copy(),'m4':aa_groups.copy()}
    transition_prob_sp = {'n1':{},'n2':{},'h1':{},'h20':{},'c10':{}} #only these have possible
                                                                # transitions to more than one state
    emission_prob_anch = {'Ah': aa_groups.copy(), 'An': aa_groups.copy(), 'An1':aa_groups.copy()}
    transition_prob_anch = {'An2':{},'Ah1':{}}

    if type == 'SP':
        emission_prob = emission_prob_sp
        transition_prob = transition_prob_sp
    else:
        emission_prob = emission_prob_anch
        transition_prob = transition_prob_anch

    for i in range(0,len(grp_sequences)):
        for j in range(0,len(grp_sequences[i])):
            if state_labels[i][j] in emission_prob.keys(): #if the state has it's own designated non-tied distribution
                                                            #e.g c1,c2,c3.. etc. see above
                region = state_labels[i][j]
            else:
                number_pos = re.search('[1-9]',state_labels[i][j]).start()
                region = state_labels[i][j][:number_pos]
            emission_prob[region][grp_sequences[i][j]] += 1
            if j < len(state_labels[i]) - 1: #No transition from the last state
                if state_labels[i][j] in transition_prob.keys():
                    if state_labels[i][j+1] in transition_prob[state_labels[i][j]].keys():
                        transition_prob[state_labels[i][j]][state_labels[i][j + 1]] += 1
                    else:
                        transition_prob[state_labels[i][j]][state_labels[i][j + 1]] = 1

    #Normalizing the distributions
    for region in emission_prob:
        if sum(emission_prob[region].values()) != 0:
            factor = 1.0/sum(emission_prob[region].values())
            for aa_emission in emission_prob[region]:
                emission_prob[region][aa_emission] = emission_prob[region][aa_emission] * factor
        else:
            print('This state has zero emissions: ', region)
    for trans_from in transition_prob:
        if sum(transition_prob[trans_from].values()) != 0:
            factor = 1.0/sum(transition_prob[trans_from].values())
            for trans_to in transition_prob[trans_from]:
                transition_prob[trans_from][trans_to] = transition_prob[trans_from][trans_to] * factor

    return emission_prob,transition_prob

def signal_peptide_specific_hmm(emission_prob, transition_prob,sig_pep_sequences_array):
    '''
    Forms the signal peptide specific hmm, where labeled data is used such that the cleavage site is always positioned
    correctly (the sequences are cut at the signal peptide cleavage site to just include the sequence of the SP and
    not the mature protein).
    '''

    #Probability distributions for emissions in regions/states
    Dn = DiscreteDistribution(emission_prob['n'])
    Dh = DiscreteDistribution(emission_prob['h'])
    Dc = DiscreteDistribution(emission_prob['c'])
    Dn1 = DiscreteDistribution(emission_prob['n1'])
    Dc1 = DiscreteDistribution(emission_prob['c1'])
    Dc2 = DiscreteDistribution(emission_prob['c2'])
    Dc3 = DiscreteDistribution(emission_prob['c3'])
    Dc4 = DiscreteDistribution(emission_prob['c4'])
    Dc5 = DiscreteDistribution(emission_prob['c5'])
    Dc6 = DiscreteDistribution(emission_prob['c6'])
    Dm1 = DiscreteDistribution(emission_prob['m1'])
    Dm2 = DiscreteDistribution(emission_prob['m2'])
    Dm3 = DiscreteDistribution(emission_prob['m3'])
    Dm4 = DiscreteDistribution(emission_prob['m4'])

    n1 = State(Dn1, name='n1')
    n2 = State(Dn, name='n2')
    n3 = State(Dn, name='n3')
    n4 = State(Dn, name='n4')
    n5 = State(Dn, name='n5')
    n6 = State(Dn, name='n6')
    n7 = State(Dn, name='n7')
    n8 = State(Dn, name='n8')
    h1 = State(Dh, name='h1')
    h2 = State(Dh, name='h2')
    h3 = State(Dh, name='h3')
    h4 = State(Dh, name='h4')
    h5 = State(Dh, name='h5')
    h6 = State(Dh, name='h6')
    h7 = State(Dh, name='h7')
    h8 = State(Dh, name='h8')
    h9 = State(Dh, name='h9')
    h10 = State(Dh, name='h10')
    h11 = State(Dh, name='h11')
    h12 = State(Dh, name='h12')
    h13 = State(Dh, name='h13')
    h14 = State(Dh, name='h14')
    h15 = State(Dh, name='h15')
    h16 = State(Dh, name='h16')
    h17 = State(Dh, name='h17')
    h18 = State(Dh, name='h18')
    h19 = State(Dh, name='h19')
    h20 = State(Dh, name='h20')
    c1 = State(Dc1, name='c1')
    c2 = State(Dc2, name='c2')
    c3 = State(Dc3, name='c3')
    c4 = State(Dc4, name='c4')
    c5 = State(Dc5, name='c5')
    c6 = State(Dc6, name='c6')
    c7 = State(Dc, name='c7')
    c8 = State(Dc, name='c8')
    c9 = State(Dc, name='c9')
    c10 = State(Dc, name='c10')
    m1 = State(Dm1, name='m1')
    m2 = State(Dm2, name='m2')
    m3 = State(Dm3, name='m3')
    m4 = State(Dm4, name='m4')

    states = [n1,n2,n3,n4,n5,n6,n7,n8,h1,h2,h3,h4,h5,h6,h7,h8,h9,
              h10,h11,h12,h13,h14,h15,h16,h17,h18,h19,h20,c10,c9,
              c8,c7,c6,c5,c4,c3,c2,c1,m1,m2,m3,m4]
    state_name_to_state_dict = {'n1': n1,'n2': n2,'n3': n3,'n4': n4,'n5': n5,'n6': n6,'n7': n7,'n8': n8,'h1': h1,
                                'h2': h2,'h3': h3,'h4': h4,'h5': h5,'h6': h6,'h7': h7,'h8': h8,'h9': h9,'h10': h10,
                                'h11': h11,'h12': h12,'h13': h13,'h14': h14,'h15': h15,'h16': h16,'h17': h17,'h18': h18,
                                'h19': h19,'h20': h20,'c1': c1,'c2': c2,'c3': c3,'c4': c4,'c5': c5,'c6': c6,'c7': c7,
                                'c8': c8,'c9': c9,'c10': c10,'m1': m1,'m2': m2,'m3': m3,'m4': m4}

    model = HiddenMarkovModel()
    model.add_states(states)

    for i in range(0,len(states)-1):
        if states[i].name in transition_prob.keys():
            for to_state in transition_prob[states[i].name]:
                model.add_transition(states[i],state_name_to_state_dict[to_state],
                                     transition_prob[states[i].name][to_state])
        else:
            model.add_transition(states[i],states[i + 1], 1)
    model.add_transition(model.start,n1,1)
    model.add_transition(m4,model.end,1)

    model.bake(merge='None')

    model.fit(sig_pep_sequences_array,distribution_inertia=0.3, edge_inertia=0.25)

    return model

def signal_anchor_hmm(emission_prob,transition_prob):

    DAn = DiscreteDistribution(emission_prob['An'])
    DAh = DiscreteDistribution(emission_prob['Ah'])
    DAn1 = DiscreteDistribution(emission_prob['An1'])

    An1 = State(DAn1, name='An1')
    An2 = State(DAn, name='An2')
    An3 = State(DAn, name='An3')
    An4 = State(DAn, name='An4')
    An5 = State(DAn, name='An5')
    An6 = State(DAn, name='An6')
    Ah1 = State(DAh, name='Ah1')
    Ah2 = State(DAh, name='Ah2')
    Ah3 = State(DAh, name='Ah3')
    Ah4 = State(DAh, name='Ah4')
    Ah5 = State(DAh, name='Ah5')
    Ah6 = State(DAh, name='Ah6')
    Ah7 = State(DAh, name='Ah7')
    Ah8 = State(DAh, name='Ah8')
    Ah9 = State(DAh, name='Ah9')
    Ah10 = State(DAh, name='Ah10')
    Ah11 = State(DAh, name='Ah11')
    Ah12 = State(DAh, name='Ah12')

    states = [An1,An2,An3,An4,An5,Ah1,Ah2,Ah3,Ah4,Ah5,Ah6,Ah7,Ah8,Ah9,Ah10,Ah11,Ah12]
    state_name_to_state_dict = {'An1': An1,'An2': An2,'An3': An3,'An4': An4,'An5': An5, 'An6': An6,'Ah1': Ah1,'Ah2': Ah2,'Ah3': Ah3,
                                'Ah4': Ah4,'Ah5': Ah5,'Ah6': Ah6,'Ah7': Ah7,'Ah8': Ah8,'Ah9': Ah9,'Ah10': Ah10,
                                'Ah11': Ah11,'Ah12': Ah12}

    model = HiddenMarkovModel('anchor')
    model.add_states(states)
    # for to_state in
    # model.add_transition('An1')
    for i in range(0, len(states) - 1):
        if states[i].name in transition_prob.keys():
            for to_state in transition_prob[states[i].name]:
                model.add_transition(states[i], state_name_to_state_dict[to_state],
                                     transition_prob[states[i].name][to_state])
        else:
            model.add_transition(states[i], states[i + 1], 1)

    model.add_transition(model.start, An1, 1)
    model.add_transition(Ah12, model.end, 1)

    model.bake(merge='None')
    return model

def emission_mature_protein(grp_sequences):
    emissions = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
    for sequence in grp_sequences:
        for aa in sequence:
            emissions[aa] += 1
    factor = 1.0/sum(emissions.values())
    for aa_emission in emissions:
        emissions[aa_emission] = emissions[aa_emission] * factor
    return emissions

def mature_protein_hmm(emission_prob):
    Dm5 = DiscreteDistribution(emission_prob)

    m5 = State(Dm5, name='m5')

    model = HiddenMarkovModel('mature')
    model.add_states(m5)

    model.add_transitions(m5, [m5,model.end], [0.99,0.01]) #number doesnt really matter as end point is given
                                                            # , end of sequence
    model.add_transition(model.start,m5,1)
    model.bake(merge='None')
    return model


def make_total_model(organism,fasta_dict,pickle_location):
    if os.path.exists(os.path.join(pickle_location, organism + '_tot_model_hmm.p')):
        tot_model = pickle.load(open(os.path.join(pickle_location, organism + '_tot_model_hmm.p'), 'rb'))
    else:
        existing_models = 0
        if os.path.exists(os.path.join(pickle_location, organism + '_sig_pep_hmm.p')):
            sigphmm = pickle.load(open(os.path.join(pickle_location, organism + '_sig_pep_hmm.p'), 'rb'))
            existing_models += 1
        if os.path.exists(os.path.join(pickle_location, organism + '_mature_hmm.p')):
            maturehmm = pickle.load(open(os.path.join(pickle_location, organism + '_mature_hmm.p'), 'rb'))
            existing_models += 1
        if organism == 'EUKARYA' and os.path.exists(os.path.join(pickle_location, organism + '_sig_anch_hmm.p')):
            anchhmm = pickle.load(open(os.path.join(pickle_location, organism + '_sig_anch_hmm.p'), 'rb'))
            existing_models += 1
        if existing_models < 3:
            sig_pep_sequences = []
            sig_anch_sequences = []
            mat_prot_sequences = []
            state_labels_sig_pep = []
            state_labels_sig_anch = []

            for sample in fasta_dict:
                if fasta_dict[sample]['type'] == 'SP' and fasta_dict[sample]['organism'] == organism \
                        and not re.search('[LT]', fasta_dict[sample]['labels']): #Exclude Tat SecI peptidase cleaved and Lipoproteins for prokaryotes
                    aa_group_sequence = amino_acid_seq_to_group_seq(fasta_dict[sample]['sequence'])
                    manual_states = manual_assignment_of_states_sp(aa_group_sequence, fasta_dict[sample]['labels'])
                    if not manual_states: #If the sample sequence doesn't fit with the manual assignment (very fex cases), ignore it
                        continue
                    state_labels_sig_pep.append(manual_states)

                    cleavage_site = re.search('[OI]', fasta_dict[sample]['labels']).start()
                    grp_seq = aa_group_sequence[:cleavage_site + 4]

                    sig_pep_sequences.append(list(grp_seq))
                elif fasta_dict[sample]['type'] == 'NO_SP' and fasta_dict[sample]['organism'] == organism:
                    if not re.search('M', fasta_dict[sample]['labels']):
                        grp_seq = amino_acid_seq_to_group_seq(fasta_dict[sample]['sequence'])

                        mat_prot_sequences.append(list(grp_seq))
                    if organism == 'EUKARYA' and fasta_dict[sample]['organism'] == organism \
                        and re.search('M', fasta_dict[sample]['labels']) \
                        and re.search('I',fasta_dict[sample]['labels']) \
                        and re.search('I',fasta_dict[sample]['labels']).start() < re.search('M', fasta_dict[sample]['labels']).start():

                        manual_states = manual_assignment_of_states_anch(fasta_dict[sample]['labels'])
                        if not manual_states:
                            continue
                        state_labels_sig_anch.append(manual_states)

                        end_anch = re.search('M', fasta_dict[sample]['labels']).end()
                        grp_seq = amino_acid_seq_to_group_seq(fasta_dict[sample]['sequence'][:end_anch])

                        sig_anch_sequences.append(list(grp_seq))
            if not os.path.exists(os.path.join(pickle_location, organism + '_sig_pep_hmm.p')):
                emission_prob_sig_pep, transition_prob_sig_pep = deduce_initial_emission_and_trans_dist(sig_pep_sequences, state_labels_sig_pep, 'SP')
                sigphmm = signal_peptide_specific_hmm(emission_prob_sig_pep, transition_prob_sig_pep, sig_pep_sequences)
                pickle.dump(sigphmm, open(os.path.join(pickle_location, organism + '_sig_pep_hmm.p'), 'wb'))

            if organism == 'EUKARYA' and not os.path.exists(os.path.join(pickle_location, organism + '_sig_anch_hmm.p')):
                emission_prob_sig_anch, transition_prob_sig_anch = deduce_initial_emission_and_trans_dist(sig_anch_sequences, state_labels_sig_anch, 'ANCH')
                anchhmm = signal_anchor_hmm(emission_prob_sig_anch,transition_prob_sig_anch)
                pickle.dump(anchhmm, open(os.path.join(pickle_location, organism + '_sig_anch_hmm.p'), 'wb'))

            if not os.path.exists(os.path.join(pickle_location, organism + '_mature_hmm.p')):
                emission_prob_mature = emission_mature_protein(mat_prot_sequences)
                maturehmm = mature_protein_hmm(emission_prob_mature)
                pickle.dump(maturehmm, open(os.path.join(pickle_location, organism + '_mature_hmm.p'), 'wb'))
        sp_hmm_states = sigphmm.states
        for state in sp_hmm_states:
            if state.name == 'n1':
                n1 = state
            elif state.name == 'm4':
                m4 = state
        transition_matrix_sig = sigphmm.dense_transition_matrix()
        # [from,to]
        if organism == 'EUKARYA':
            for state in anchhmm.states:
                if state.name == 'An1':
                    An1 = state
                elif state.name == 'Ah12':
                    Ah12 = state
        for state in maturehmm.states:
            if state.name == 'm5':
                m5_1 = state.copy()
                m5_1.name = 'm5_1'
                m5_2 = state.copy()
                m5_2.name = 'm5_2'
                m5_3 = state.copy()
                m5_3.name = 'm5_3'
        tot_model = HiddenMarkovModel()
        tot_model.add_states([m5_1,m5_2])
        tot_model.add_states(sp_hmm_states)
        for i in range(len(sp_hmm_states)):
            for j in range(len(sp_hmm_states)):
                if transition_matrix_sig[i,j] != 0:
                    tot_model.add_transition(sp_hmm_states[i],sp_hmm_states[j],transition_matrix_sig[i,j], pseudocount = 1e08)
                    #the high pseudocount ensures that these edges are not updated during training of the total model.
        # tot_model.add_model(sigphmm)
        if organism == 'EUKARYA':
            transition_matrix_anch = anchhmm.dense_transition_matrix()
            anch_hmm_states = anchhmm.states
            tot_model.add_states(anch_hmm_states)
            for i in range(len(anch_hmm_states)):
                for j in range(len(anch_hmm_states)):
                    if transition_matrix_anch[i,j] != 0:
                        tot_model.add_transition(anch_hmm_states[i],anch_hmm_states[j],transition_matrix_anch[i,j], pseudocount = 1e08)
            tot_model.add_state(m5_3)
            tot_model.add_transition(m5_3,m5_3,0.99)
            tot_model.add_transition(Ah12, m5_3, 1)
            tot_model.add_transition(tot_model.start, An1, 0.1)
            tot_model.add_transition(m5_3, tot_model.end, 0.01)
        tot_model.add_transition(tot_model.start, m5_1, 0.95)
        tot_model.add_transition(m5_1,tot_model.end,0.01)
        tot_model.add_transition(m5_2, tot_model.end, 0.01)
        tot_model.add_transitions([m5_1,m5_2],[m5_1,m5_2],[0.99,0.99])
        tot_model.add_transition(tot_model.start,n1,0.05)
        tot_model.add_transition(m4, m5_2, 1)
        tot_model.bake()

        grp_sequences = []
        types_of_interest = ['SP','NO_SP']
        for sample in fasta_dict:
            if fasta_dict[sample]['organism'] == organism and fasta_dict[sample]['type'] in types_of_interest:
                aa_group_sequence = amino_acid_seq_to_group_seq(fasta_dict[sample]['sequence'])
                grp_sequences.append(list(aa_group_sequence))
        tot_model.freeze_distributions() #Freezes all distribution parameters of the model, so that they don't update during training, have already been trained in the separate models
        tot_model.fit(grp_sequences,use_pseudocount = True, edge_inertia = 0.5) #By using the high pseudocounts set on all edges from the signal peptide and anchor hmm, these remain unchanged during training.
        pickle.dump(tot_model, open(os.path.join(pickle_location, organism + '_tot_model_hmm.p'), 'wb'))

    return tot_model

def score_prediction(fasta_dict, organism, model):
    cleavage_correct = [0,0,0,0]
    predicted_type = []
    actual_type = []
    sp_correct = 0
    sp_incorrect = 0
    no_sp_incorrect = 0
    no_sp_correct = 0
    for sample in fasta_dict:
        if fasta_dict[sample]['organism'] == organism:
            if fasta_dict[sample]['type'] == 'SP' or fasta_dict[sample]['type'] == 'NO_SP':
                grp_seq = amino_acid_seq_to_group_seq(fasta_dict[sample]['sequence'])
                log_p, path = model.viterbi(list(grp_seq))
                prediction = [state.name for idx, state in path[1:-1]] #path is a list of the predicted states for each observation, but also includes the end
                                        #state as well as the start state, so the actual prediction is the list between the
                                        #last and first element in path
                if fasta_dict[sample]['type'] == 'SP':
                    actual_type.append(1)
                if 'n1' in prediction:  # If it is predicted as containing an sp, therefore containing the n1 state
                    predicted_type.append(1)
                    if fasta_dict[sample]['type'] == 'SP':
                        sp_correct += 1
                        predicted_cleavage = prediction.index('c1')
                        actual_cleavage = re.search('S+',fasta_dict[sample]['labels']).end() - 1
                        if abs(predicted_cleavage - actual_cleavage) <= 3:
                            cleavage_correct[abs(predicted_cleavage - actual_cleavage)] += 1
                    else:
                        sp_incorrect += 1
                if fasta_dict[sample]['type'] == 'NO_SP':
                    actual_type.append(0)
                if 'm5_1' in prediction or 'An1' in prediction:
                    predicted_type.append(0)
                    if fasta_dict[sample]['type'] == 'NO_SP':
                        no_sp_correct += 1
                    else:
                        no_sp_incorrect += 1
    print(len(actual_type))
    auc_score = sk_met.roc_auc_score(actual_type,predicted_type)
    mcc_score = sk_met.matthews_corrcoef(actual_type,predicted_type)
    cs_recall = [100*sum(cleavage_correct[:x+1])/ (no_sp_incorrect + sum(cleavage_correct[:x+1])) for x in range(len(cleavage_correct))]
    cs_precision = [100*sum(cleavage_correct[:x+1])/ (sp_incorrect + sp_correct) for x in range(len(cleavage_correct))]
    return auc_score,mcc_score, cs_recall, cs_precision

def model_and_predict_all_organisms(fasta_dict_train,fasta_dict_test,pickle_location,organisms):
    auc_scores = {}
    mcc_scores = {}
    cs_recalls = {}
    cs_precisions = {}
    for organism in organisms:
        hmm = make_total_model(organism,fasta_dict_train,pickle_location)
        auc_score, mcc_score, cs_recall,cs_precision = score_prediction(fasta_dict_test, organism, hmm)
        auc_scores[organism] = auc_score
        mcc_scores[organism] = mcc_score
        cs_recalls[organism] = cs_recall
        cs_precisions[organism] = cs_precision
    return auc_scores, mcc_scores, cs_recalls,cs_precisions

def five_fold_cross_validation(filename_data_set,pickle_location):
    organisms_of_interest = ['EUKARYA','NEGATIVE','POSITIVE']
    types_of_interest = ['SP','NO_SP']
    total_fasta_dict = fasta_file_to_dict(filename_data_set)
    clustered_total_fasta_dict = {}
    for sample in total_fasta_dict:
        organism = total_fasta_dict[sample]['organism']
        type = total_fasta_dict[sample]['type']
        if organism in organisms_of_interest:
            if organism not in clustered_total_fasta_dict.keys():
                clustered_total_fasta_dict[organism] = {}
            if type in types_of_interest:
                if type not in clustered_total_fasta_dict[organism]:
                    clustered_total_fasta_dict[organism][type] = {}
                clustered_total_fasta_dict[organism][type][sample] = total_fasta_dict[sample]

    uniprot_acs_clustered = {}
    number_of_samples_per_category = {}
    for organism in organisms_of_interest:
        uniprot_acs_clustered[organism] = {}
        number_of_samples_per_category[organism] = {}
        for type in types_of_interest:
            uniprot_acs_clustered[organism][type] = list(clustered_total_fasta_dict[organism][type].keys())
            number_of_samples_per_category[organism][type] = len(uniprot_acs_clustered[organism][type])

    cross_val_set1 = {}
    cross_val_set2 = {}
    cross_val_set3 = {}
    cross_val_set4 = {}
    cross_val_set5 = {}
    cross_val_sets = [cross_val_set1, cross_val_set2, cross_val_set3, cross_val_set4, cross_val_set5]

    for organism in organisms_of_interest:
        for type in types_of_interest:
            category_dict = clustered_total_fasta_dict[organism][type]
            number_in_category = number_of_samples_per_category[organism][type]
            num_per_set = math.floor(number_in_category/5)
            cross_val_set1.update(dict(list(category_dict.items())[:num_per_set]))
            cross_val_set2.update(dict(list(category_dict.items())[num_per_set:num_per_set*2]))
            cross_val_set3.update(dict(list(category_dict.items())[num_per_set*2:num_per_set*3]))
            cross_val_set4.update(dict(list(category_dict.items())[num_per_set*3:num_per_set*4]))
            cross_val_set5.update(dict(list(category_dict.items())[num_per_set*4:]))

    #Shuffle all the entries in each set
    for i in range(len(cross_val_sets)):
        uniprot_acs = list(cross_val_sets[i].keys())
        shuffled_set = {}
        np.random.shuffle(uniprot_acs)
        for uniprot_ac in uniprot_acs:
            shuffled_set[uniprot_ac] = total_fasta_dict[uniprot_ac]
        cross_val_sets[i] = shuffled_set

    auc_scores_by_organism = {}
    mcc_scores_by_organism = {}
    cs_recalls_by_organism = {}
    cs_precisions_by_organism = {}

    for i in range(len(cross_val_sets)):
        pickle_location_cross = pickle_location + '_cross_' + str(i+1)
        test_dict = cross_val_sets[i]
        train_dict = {}
        for j in range(len(cross_val_sets)):
            if j != i:
                train_dict.update(cross_val_sets[j])
        print(len(train_dict))
        auc_scores, mcc_scores, cs_recalls,cs_precisions = model_and_predict_all_organisms(train_dict,test_dict,pickle_location_cross,organisms_of_interest)
        for organism in organisms_of_interest:
            if organism not in auc_scores_by_organism.keys():
                auc_scores_by_organism[organism] = []
                mcc_scores_by_organism[organism] = []
                cs_recalls_by_organism[organism] = []
                cs_precisions_by_organism[organism] = []
            auc_scores_by_organism[organism].append(auc_scores[organism])
            mcc_scores_by_organism[organism].append(mcc_scores[organism])
            cs_recalls_by_organism[organism].append(cs_recalls[organism])
            cs_precisions_by_organism[organism].append(cs_precisions[organism])

    for organism in organisms_of_interest:
        av_auc_score = sum(auc_scores_by_organism[organism]) /5
        av_mcc_score = sum(mcc_scores_by_organism[organism]) / 5
        av_cs_recall = [sum(np.array(cs_recalls_by_organism[organism])[:,x]) / 5 for x in range(4)]
        av_cs_precision = [sum(np.array(cs_precisions_by_organism[organism])[:, x]) / 5 for x in range(4)]
        print('AUC scores for ', organism, ': ', auc_scores_by_organism[organism], 'average: ', av_auc_score)
        print('MCC scores for ', organism, ': ', mcc_scores_by_organism[organism], 'average: ', av_mcc_score)
        print('CS recall scores for ', organism, ': ', cs_recalls_by_organism[organism], 'average: ', av_cs_recall)
        print('CS precision scores for ', organism, ': ', cs_precisions_by_organism[organism], 'average: ', av_cs_precision)



if __name__ == '__main__':
    pickle_locations = 'hmm_pickles'
    dataset = 'benchmark_set.fasta'
    five_fold_cross_validation(dataset,pickle_locations)
