def update(word, predict_cmty, model, _alpha, _work):
    cdef np.uint32_t *word_point
    cdef np.uint8_t *word_code
    cdef int codelen
    if word is None:
        codelen = 0
    else:
        word_point = <np.uint32_t *>np.PyArray_DATA(word.point)
        word_code = <np.uint8_t *>np.PyArray_DATA(word.code)
        codelen = <int>len(word.code)

    cdef long long a, b
    cdef np.uint32_t topic_index = <np.uint32_t> predict_cmty
    cdef long long row1 = topic_index * size, row2
    cdef int size = model.layer1_size
    cdef REAL_t *syn0_topic = <REAL_t *> (np.PyArray_DATA(model.syn0_topic))
    cdef REAL_t *syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))
    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *work = <REAL_t *>np.PyArray_DATA(_work)
    cdef REAL_t f, g
    cdef REAL_t alpha = <REAL_t>_alpha
    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f = <REAL_t>sdot(&size, &syn0_topic[row1], &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &syn0_topic[row1], &ONE, &syn1[row2], &ONE)
    saxpy(&size, &ONEF, work, &ONE, &syn0_topic[row1], &ONE)
