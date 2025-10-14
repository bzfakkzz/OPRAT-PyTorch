

def CountInSeq(S,C,attacks):
    count_min=1000
    for j in C:
        count_min=min(count_min,S.count(attacks[j]))

    F_min=[]
    c=[]
    for i,attack in enumerate(attacks):
        if S.count(attack)==count_min:
            F_min.append(attack)
            c.append(i)

    return F_min,c