import flask
from flask import request,jsonify
import numpy as np
# importing Qiskit
from qiskit.providers.ibmq import least_busy
# backend = least_busy(IBMQ.backends(filters=lambda x: not x.configuration().simulator))
# backend.name()
from qiskit import Aer,BasicAer,IBMQ
from qiskit import QuantumCircuit, assemble, execute,ClassicalRegister,transpile, QuantumRegister
# import basic plot tools
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from random import choice
import io
import json
import base64
from qiskit.circuit import qpy_serialization
from qiskit.aqua.components.oracles import TruthTableOracle
from qiskit.tools.monitor import job_monitor
import operator


def gaussian_elimination(msr):
    # print(msr)
    lst = [(k[::-1], v) for k, v in msr.items() if k != "0" * len(k)]
    lst.sort(key=lambda x: x[1], reverse=True)
    n = len(lst[0][0])
    eqn = []
    for k, _ in lst:
        eqn.append([int(c) for c in k])
    y = Matrix(eqn)
    yt = y.rref(iszerofunc=lambda x: x % 2 == 0)

    def mod(x, modulus):
        num, den = x.as_numer_denom()
        return num * mod_inverse(den, modulus) % modulus

    y_new = yt[0].applyfunc(lambda x: mod(x, 2))
    rows, _ = y_new.shape
    hidden = [0] * n
    for r in range(rows):
        yi = [i for i, v in enumerate(list(y_new[r, :])) if v == 1]
        if len(yi) == 2:
            hidden[yi[0]] = '1'
            hidden[yi[1]] = '1'
    key = "".join(str(h) for h in hidden)[::-1]
    return key

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return jsonify({'response':'hello'})


@app.route('/demo/get_oracle',methods=['GET'])
def build_oracle():
    if 'n' in request.args:
        n = int(request.args['n'])
    else:
        return "Please specify number of qubits"

    if 'case' in request.args:
        case = str(request.args['case'])
    else:
        return "Please specify type of oracle"
    # if case == None:
    #     case = choice(['b', 'c'])

    oracle = QuantumCircuit(n + 1, n + 1)
    for i in range(0, n):
        oracle.h(i)

    oracle.x(n)
    oracle.h(n)

    oracle.barrier()
    if case == 'b':
        b = np.random.randint(0, pow(2, n) - 1)
        bitmap = format(b, '0' + str(n) + 'b')
        # print(f'Bit String for input: {bitmap}')
        for qbit in range(len(bitmap)):
            if bitmap[qbit] == '1':
                oracle.x(qbit)
        # oracle.barrier()
        for qbit in range(n):
            oracle.cx(qbit, n)
        # oracle.barrier()
        for qbit in range(len(bitmap)):
            if bitmap[qbit] == '1':
                oracle.x(qbit)
    elif case == 'c':
        output = np.random.randint(2)
        case += str(output)
        print(output)
        if output == 0:
            oracle.z(n)
    oracle.barrier()

    if 'measure' in request.args:
        if str(request.args['measure']) == 'True':
            for q in range(n + 1):
                oracle.h(q)
                oracle.measure(q, q)

    buf = io.BytesIO()
    qpy_serialization.dump(oracle, buf)
    json_str = json.dumps({
        'oracle': base64.b64encode(buf.getvalue()).decode('utf8')
    })

    return json_str


@app.route('/demo/get_type',methods=['GET'])

def get_type():
    if 'circuit' in request.args:
        # circuit_json = str(json.loads(request.args['circuit']))
        circuit_json = request.args['circuit']
        qpy_file = io.BytesIO(base64.b64decode(circuit_json))
        circuit = qpy_serialization.load(qpy_file)[0]
    else:
        return "Please provide circuit."

    aer_sim = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, aer_sim, shots=1024, memory=True)
    result = job.result()
    measurements = result.get_memory()[0]
    # msr = result.get_memory()
    # c1 = len(list(x for x in msr if x[0] == '0'))
    # c2 = len(list(x for x in msr if x[0] == '1'))
    # print(f'0:{c1},1:{c2}')
    # print(measurements)
    query_state = measurements[-1]
    if query_state == '1':
        # print(aux_state)
        type = 'BALANCED'
    else:  # constant query
        aux_output = measurements[0]
        if aux_output == '1':
            type = 'Constantly 1'
        else:
            type = 'Constantly 0'
    resp = {'type' : type}
    return jsonify(resp)

#
# @app.route('/d_josza',methods=['GET'])
# def

@app.route('/d_josza',methods=['GET'])
def D_Josza():
    if 'bitmap' in request.args:
        bitmap = str(request.args['bitmap'])
        n = int(np.log2(len(bitmap)))
        if len(bitmap) != pow(2, n):
            return 'Length of bitmap should be in powers of 2'
    else:
        return 'No Bitmap provided'
    if 'key' in request.args:
        API_KEY = str(request.args['key'])
    else:
        return 'No IBMQ API key found.'
    oracle = TruthTableOracle(bitmap, optimization=True, mct_mode='noancilla')
    pre = QuantumCircuit(oracle.variable_register, oracle.output_register)
    pre.h(oracle.variable_register)
    pre.x(oracle.output_register)
    pre.h(oracle.output_register)
    mid = oracle.construct_circuit()
    post = QuantumCircuit(oracle.variable_register, oracle.output_register)
    post.h(oracle.variable_register)
    circuit = mid.compose(pre,front=True).compose(post)
    res = ''
    msr = ClassicalRegister(oracle.variable_register.size)
    circuit.add_register(msr)
    # msr2 = ClassicalRegister(reg_out.size)
    # circuit.add_register(msr2)
    circuit.measure(oracle.variable_register, msr)
    # circuit.measure(reg_out,msr2)
    # backend = Aer.get_backend('qasm_simulator')
    # job = execute(circuit, backend, shots=1)
    # result = job.result()
    # m = result.get_counts()

    # API_KEY = ''
    # IBMQ.save_account(API_KEY)
#     IBMQ.disable_account()
    IBMQ.enable_account(API_KEY)
    provider = IBMQ.get_provider('ibm-q')
    backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= (n + 1) and
                                           not x.configuration().simulator and
                                           x.status().operational == True))
    # backend = provider.get_backend('ibmq_5_yorktown')
    job = execute(circuit, backend, shots=1024)
    result = job.result()
    m = result.get_counts()

    # aer_sim = Aer.get_backend('aer_simulator')
    # qobj = assemble(circuit, aer_sim)
    # results = aer_sim.run(qobj).result()
    # m = results.get_counts()

        # print(m)
    top_measurement = max(m.items(), key=operator.itemgetter(1))[0]
    print(top_measurement)
    query_state = int(top_measurement)
    if query_state == 0:
        res = 'constant'
    else: res = 'balanced'
    json_str = {'type': res}
        # buf = io.BytesIO()
        # qpy_serialization.dump(circuit, buf)
        # json_str = json.dumps({
        #     'dj_oracle': base64.b64encode(buf.getvalue()).decode('utf8'),
        #     'qubits': str(n),
        #     'bmp': bitmap
        # })
        # json_str = json.dumps(assemble(circuit).to_dict())
        # resp = {'dj_oracle': json_str}
    IBMQ.disable_account()
    return jsonify(json_str)


@app.route('/demo/get_BV_oracle',methods=['GET'])
def build_BV_oracle():
    if 'key' in request.args:
        key = request.args['key']
        n = len(key)
    elif 'qubits' in request.args:
        n = int(request.args['qubits'])
        key = np.random.randint(0, pow(2, n) - 1)
        key = format(key, '0' + str(n) + 'b')
    else:
        return jsonify({'ERROR': 'Cannot specify key bitstring'})
    oracle = QuantumCircuit(n + 1, n)
    for i in range(n):
        oracle.h(i)
    oracle.x(n)
    oracle.h(n)
    oracle.barrier()
    for i, v in enumerate(key):
        if v == '1':
            oracle.cx(i, n)
    oracle.barrier()
    if 'measure' in request.args and request.args['measure']=='True':
        for i in range(n):
            oracle.h(i)
            oracle.measure(i, i)
    buf = io.BytesIO()
    qpy_serialization.dump(oracle, buf)
    json_str = json.dumps({
        'oracle': base64.b64encode(buf.getvalue()).decode('utf8'),
        'key': key
    })
    return json_str


@app.route('/demo/get_BV_key',methods=['GET'])
def get_key_():
    if 'oracle' in request.args:
        circuit_json = request.args['oracle']
        qpy_file = io.BytesIO(base64.b64decode(circuit_json))
        circuit = qpy_serialization.load(qpy_file)[0]
    else:
        return jsonify({'ERROR': 'No Oracle circuit found.'})
    simulator = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1, memory=True)
    result = job.result()
    measurement = result.get_memory()[0]
    measurement = measurement[::-1]
    return jsonify({'key': measurement})


@app.route('/BVazirani',methods=['GET'])
def apply_bv():
    if 'bitmap' in request.args:
        bitmap = request.args['bitmap']
        n = int(np.log2(len(bitmap)))
        if len(bitmap) != pow(2, n):
            return jsonify({'ERROR': 'bitmap length should be in powers of 2.'})
    else:
        return jsonify({'ERROR': 'Please provide bitmap.'})
    if 'api_key' in request.args:
        API_KEY = request.args['api_key']
    else:
        return jsonify({'ERROR': 'No IBM-Q API key found.'})

    oracle = TruthTableOracle(bitmap, optimization=True, mct_mode='noancilla')
    superpos = QuantumCircuit(oracle.variable_register,oracle.output_register)
    superpos.h(oracle.variable_register)
    superpos.x(oracle.output_register)
    superpos.h(oracle.output_register)
    circuit = oracle.construct_circuit()
    circuit = circuit.compose(superpos, front=True)
    desup = QuantumCircuit(oracle.variable_register,oracle.output_register)
    desup.h(oracle.variable_register)
    circuit = circuit.compose(desup)
    msr = ClassicalRegister(oracle.variable_register.size)
    circuit.add_register(msr)
    circuit.measure(oracle.variable_register,msr)

    IBMQ.enable_account(API_KEY)
    provider = IBMQ.get_provider('ibm-q')
    backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= (n + 1) and
                                           not x.configuration().simulator and
                                           x.status().operational == True))
        # backend = provider.get_backend('ibmq_5_yorktown')
#     simulator = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=1024)
    job_monitor(job)
    result = job.result()
    noisy_keys = result.get_counts()
    key = max(noisy_keys.items(), key=operator.itemgetter(1))[0]
    IBMQ.disable_account()
    return jsonify({'key': key})

@app.route('/demo/get_simon_oracle',methods=['GET'])
def get_oracle():

    if 'key' in request.args:
        b = request.args['key']
        n = len(b)
        ones = []
        for i in range(n):
            if b[i] == '1':
                ones.append(i)
    else:
        return jsonify({'ERROR': 'Cannot specify the key bitstring.'})
    qr1 = QuantumRegister(n,'reg1')
    qr2 = QuantumRegister(n,'reg2')
    mr = ClassicalRegister(n)
    orcle = QuantumCircuit(qr1,qr2,mr)

    orcle.h(qr1)
    orcle.barrier()
    orcle.cx(qr1,qr2)
    if ones:
        x = n - ones[-1] - 1
        # x = ones[0]
        for one in ones:
            orcle.cx(qr1[x], qr2[n-one-1])
            # orcle.cx(qr1[x], qr2[one])
    orcle.barrier()

    orcle.measure(qr2,mr)
    orcle.barrier()

    orcle.h(qr1)
    orcle.measure(qr1,mr)
    # orcle = QuantumCircuit(n * 2, n)
    # orcle.h(range(n))
    # orcle.barrier()
    # orcle += simon_oracle(b)
    # orcle.barrier()
    # orcle.h(range(n))
    # orcle.measure(range(n), range(n))
    buf = io.BytesIO()
    qpy_serialization.dump(orcle, buf)
    json_str = json.dumps({
        'oracle': base64.b64encode(buf.getvalue()).decode('utf8'),
        'key': b
    })
    return json_str

@app.route('/demo/get_simon_key',methods=['GET'])
def getsimonkey():
    if 'circuit' in request.args:
        orcle_json = request.args['circuit']
        qpy_file = io.BytesIO(base64.b64decode(orcle_json))
        orcle = qpy_serialization.load(qpy_file)[0]
    else:
        return jsonify({'ERROR':'No circuit provided.'})

    simulator = BasicAer.get_backend('qasm_simulator')
    job = execute(orcle, simulator, shots=1024, memory=True)
    result = job.result()
    msr = result.get_counts()
    # print(msr)
    lst = [(k[::-1], v) for k, v in msr.items() if k != "0" * len(k)]
    lst.sort(key=lambda x: x[1], reverse=True)
    n = len(lst[0][0])
    eqn = []
    for k, _ in lst:
        eqn.append([int(c) for c in k])
    y = Matrix(eqn)
    yt = y.rref(iszerofunc=lambda x: x % 2 == 0)

    def mod(x, modulus):
        num, den = x.as_numer_denom()
        return num * mod_inverse(den, modulus) % modulus

    y_new = yt[0].applyfunc(lambda x: mod(x, 2))
    rows, _ = y_new.shape
    hidden = [0] * n
    for r in range(rows):
        yi = [i for i, v in enumerate(list(y_new[r, :])) if v == 1]
        if len(yi) == 2:
            hidden[yi[0]] = '1'
            hidden[yi[1]] = '1'
    key = "".join(str(h) for h in hidden)[::-1]
    return jsonify({'key': key})

@app.route('/Simon',methods=['GET'])
def apply_simon():
    print(request.args.getlist('bitmap'))
    if 'bitmap' in request.args:
        bmp = request.args.getlist('bitmap')
        n = len(bmp[0])
        print(bmp)
        for b in bmp:
            if len(b) != n:
                return jsonify({'ERROR': 'Unequal length of bitmap outputs.'})
    else:
        return jsonify({'ERROR': 'Bitmap not  provided.'})
    if 'key' in request.args:
        API_KEY = request.args['key']
    else:
        return jsonify({'ERROR':'IBM-Q Quantum Experience key not provided.'})
    oracle = TruthTableOracle(bmp, optimization=True, mct_mode='noancilla')
    orcle = oracle.construct_circuit()
    circuit = QuantumCircuit(*orcle.qregs)
    # buf = io.BytesIO()
    # qpy_serialization.dump(circuit, buf)
    # json_str = json.dumps({
    #     'oracle': base64.b64encode(buf.getvalue()).decode('utf8')
    # })
    # return json_str
    circuit.h(oracle.variable_register)
    circuit.compose(orcle, inplace=True)
    circuit.h(oracle.variable_register)
    msr = ClassicalRegister(oracle.variable_register.size)
    circuit.add_register(msr)
    circuit.measure(oracle.variable_register, msr)

    provider = IBMQ.enable_account(API_KEY)
    # provider = IBMQ.get_provider('ibm-q')
    backend = least_busy(backends=provider.backends(filters=lambda x: x.configuration().n_qubits >= int(math.log2(n)) and
                                           not x.configuration().simulator and
                                           x.status().operational is True))
    # print(provider.backends())
    # backend = provider.get_backend('ibmq_lima')


    # simulator = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=1024)
    job_monitor(job)
    result = job.result()
    measurements = result.get_counts()
    key = gaussian_elimination(measurements)
    IBMQ.disable_account()
    return jsonify({'key': key})



if __name__=='__main__':
    app.run()

