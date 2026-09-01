const aimet = {};

aimet.EncodingFile = class {

    constructor() {
        this._reset();
    }

    get empty() {
        return this._entries.size === 0;
    }

    get version() {
        return this._version;
    }

    get entries() {
        return Array.from(this._entries.values());
    }

    get issues() {
        return this._issues;
    }

    get summary() {
        const entries = this.entries;
        const activations = entries.filter((entry) => entry.category === 'activation').length;
        const parameters = entries.filter((entry) => entry.category === 'parameter').length;
        return {
            total: entries.length,
            activations,
            parameters,
            matched: this._matched,
            unmatched: entries.length - this._matched,
            modelValues: this._modelValues,
            unencoded: Math.max(0, this._modelValues - this._matched),
            inferred: this._inferred.size
        };
    }

    get profile() {
        const entries = this.entries;
        const count = (values) => {
            const counts = new Map();
            for (const entry of values) {
                counts.set(entry.label, (counts.get(entry.label) || 0) + 1);
            }
            return Array.from(counts.entries()).sort((a, b) => a[0].localeCompare(b[0]));
        };
        const attentionLayers = new Set();
        for (const entry of entries) {
            const match = entry.name.match(/^single\.(\d+)\.value$/);
            if (match) {
                attentionLayers.add(match[1]);
            }
        }
        const cache = entries.filter((entry) => {
            if (/^past_key_values\.attention\.\d+\.(key|value)$/.test(entry.name)) {
                return true;
            }
            if (/\/self_attn\/(key_cache_concat|value_cache_concat)\//.test(entry.name)) {
                return true;
            }
            const match = entry.name.match(/^single\.(\d+)\.(key|value)$/);
            return match ? attentionLayers.has(match[1]) : false;
        });
        return {
            encodings: count(entries),
            cache: count(cache)
        };
    }

    open(data) {
        this._reset();
        if (!aimet.EncodingFile.match(data)) {
            return false;
        }
        this._version = typeof data.version === 'string' ? data.version : 'unknown';
        const major = Number.parseInt(this._version.split('.')[0], 10);
        if (major >= 2) {
            this._openList(data.activation_encodings, 'activation', (item) => this._version2(item));
            this._openList(data.param_encodings, 'parameter', (item) => this._version2(item));
        } else if (major === 1) {
            this._openList(data.activation_encodings, 'activation', (item) => this._version1(item));
            this._openList(data.param_encodings, 'parameter', (item) => this._version1(item));
        } else {
            this._openLegacy(data.activation_encodings, 'activation');
            this._openLegacy(data.param_encodings, 'parameter');
        }
        if (this._entries.size === 0) {
            this._issues.push({ severity: 'error', source: 'parse', message: 'The file does not contain any valid quantization encodings.' });
        }
        return true;
    }

    bind(model) {
        const values = new Map();
        const nodes = [];
        const visitValue = (value) => {
            if (value && typeof value.name === 'string' && value.name) {
                values.set(value.name.split('\n').shift(), value);
            }
        };
        const visitArgument = (argument) => {
            if (argument && Array.isArray(argument.value)) {
                for (const value of argument.value) {
                    visitValue(value);
                }
            }
        };
        const visitTarget = (target) => {
            for (const argument of target.inputs || []) {
                visitArgument(argument);
            }
            for (const argument of target.outputs || []) {
                visitArgument(argument);
            }
            for (const node of target.nodes || []) {
                nodes.push(node);
                for (const argument of node.inputs || []) {
                    visitArgument(argument);
                }
                for (const argument of node.outputs || []) {
                    visitArgument(argument);
                }
                for (const argument of node.blocks || []) {
                    if (argument && argument.type === 'graph' && argument.value) {
                        visitTarget(argument.value);
                    }
                }
            }
        };
        const targets = new Set([...(model.modules || []), ...(model.functions || [])]);
        for (const target of targets) {
            visitTarget(target);
        }
        this._matched = 0;
        this._modelValues = values.size;
        this._inferred = new Map();
        this._issues = this._issues.filter((issue) => issue.source === 'parse');
        for (const entry of this._entries.values()) {
            const value = values.get(entry.name);
            entry.matched = Boolean(value);
            if (!value) {
                this._issues.push({
                    severity: 'warning',
                    name: entry.name,
                    message: `Encoding tensor '${entry.name}' was not found in the model.`
                });
                continue;
            }
            this._matched++;
            const dimensions = value.type && value.type.shape && Array.isArray(value.type.shape.dimensions) ? value.type.shape.dimensions : null;
            if (entry.axis !== null && dimensions) {
                let axis = entry.axis;
                axis = axis < 0 ? dimensions.length + axis : axis;
                if (axis < 0 || axis >= dimensions.length) {
                    this._issues.push({
                        severity: 'warning',
                        name: entry.name,
                        message: `Encoding axis ${entry.axis} is outside tensor rank ${dimensions.length} for '${entry.name}'.`
                    });
                } else if (entry.granularity === 'per-channel' && entry.blockSize === null) {
                    const expected = dimensions[axis];
                    const actual = aimet.Utility.count(entry.scale);
                    if (Number.isInteger(expected) && actual > 1 && actual !== expected) {
                        this._issues.push({
                            severity: 'warning',
                            name: entry.name,
                            message: `Encoding scale count ${actual} does not match dimension ${expected} on axis ${entry.axis} for '${entry.name}'.`
                        });
                    }
                }
            }
        }
        this._infer(nodes);
    }

    get(name) {
        name = typeof name === 'string' ? name.split('\n').shift() : '';
        return this._entries.get(name) || null;
    }

    value(value) {
        return value ? this.get(value.name) : null;
    }

    precision(value) {
        if (!value) {
            return null;
        }
        const name = typeof value.name === 'string' ? value.name.split('\n').shift() : '';
        const entry = this.get(name);
        return entry && entry.category === 'activation' ? entry : (this._inferred.get(name) || null);
    }

    tensor(tensor, value) {
        let name = '';
        if (tensor && tensor.name) {
            name = tensor.name;
        } else if (value && value.name) {
            name = value.name;
        }
        return this.get(name);
    }

    node(node, inferred = true) {
        const collect = (items, category) => {
            const entries = new Map();
            for (const argument of items || []) {
                let values = argument.value;
                if (!Array.isArray(values)) {
                    values = values === undefined || values === null ? [] : [values];
                }
                for (const value of values) {
                    const entry = category === 'activation' && inferred ? this.precision(value) : this.value(value);
                    if (entry && (!category || entry.category === category)) {
                        entries.set(entry.name, entry);
                    }
                }
            }
            return Array.from(entries.values());
        };
        const inputs = collect(node.inputs, 'activation');
        const parameters = collect(node.inputs, 'parameter');
        const outputs = collect(node.outputs, 'activation');
        const entries = new Map();
        for (const entry of [...inputs, ...parameters, ...outputs]) {
            entries.set(entry.name, entry);
        }
        return {
            inputs,
            parameters,
            outputs,
            entries: Array.from(entries.values())
        };
    }

    static labels(group) {
        const labels = [];
        const append = (prefix, entries) => {
            const values = Array.from(new Set(entries.map((entry) => entry.label)))
                .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
            if (values.length > 0) {
                labels.push(`${prefix}:${values.join('/')}`);
            }
        };
        append('I', group.inputs);
        append('P', group.parameters);
        append('O', group.outputs);
        return labels;
    }

    static description(group) {
        const labels = [];
        const append = (name, entries) => {
            const values = Array.from(new Set(entries.map((entry) => entry.label)))
                .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
                .map((label) => {
                    const matching = entries.filter((entry) => entry.label === label);
                    const inferred = matching.filter((entry) => entry.inferred).length;
                    if (inferred === matching.length) {
                        return `${label} (inferred)`;
                    }
                    return inferred > 0 ? `${label} (partly inferred)` : label;
                });
            if (values.length > 0) {
                labels.push(`${name}: ${values.join('/')}`);
            }
        };
        append('Input', group.inputs);
        append('Parameter', group.parameters);
        append('Output', group.outputs);
        return labels;
    }

    static nodeBadge(explicit, precision) {
        const unique = (entries) => Array.from(new Set(entries.map((entry) => entry.label)))
            .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
        const labels = [];
        const descriptions = [];
        const transition = aimet.EncodingFile.transition(precision);
        if (transition && explicit.outputs.length > 0) {
            labels.push(transition.replaceAll(' ', ''));
            descriptions.push(`Quantization signature: ${transition} (explicit output QParam)`);
        } else {
            const outputs = unique(explicit.outputs);
            if (outputs.length > 0) {
                labels.push(outputs.join('/'));
                descriptions.push(`Output QParams: ${outputs.join('/')}`);
            }
        }
        const parameters = unique(explicit.parameters);
        if (parameters.length > 0) {
            labels.push(parameters.join('/'));
            descriptions.push(`Parameter QParams: ${parameters.join('/')}`);
        }
        return {
            labels,
            descriptions
        };
    }

    static precision(group) {
        const activations = [...group.inputs, ...group.outputs];
        const labels = new Set(activations.map((entry) => entry.label));
        if (labels.size > 1) {
            return 'mixed';
        }
        if (labels.size === 1) {
            return Array.from(labels)[0].toLowerCase();
        }
        const parameters = new Set(group.parameters.map((entry) => entry.label));
        return parameters.size === 1 ? Array.from(parameters)[0].toLowerCase() : 'mixed';
    }

    static transition(group) {
        const unique = (entries) => Array.from(new Set(entries.map((entry) => entry.label)))
            .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
        const inputs = unique(group.inputs);
        const outputs = unique(group.outputs);
        if (inputs.length > 0 && outputs.length === 1) {
            const input = inputs.join('/');
            const output = outputs[0];
            return inputs.length === 1 && input === output ? '' : `${input} \u2192 ${output}`;
        }
        return '';
    }

    static match(data) {
        return data && typeof data === 'object' && typeof data.version === 'string' &&
            ('activation_encodings' in data || 'param_encodings' in data) &&
            data.signature !== 'netron:attachment';
    }

    _reset() {
        this._version = '';
        this._entries = new Map();
        this._issues = [];
        this._matched = 0;
        this._modelValues = 0;
        this._inferred = new Map();
    }

    _infer(nodes) {
        const passthrough = new Set([
            'Identity',
            'Cast',
            'Transpose',
            'Reshape',
            'Flatten',
            'Squeeze',
            'Unsqueeze',
            'DepthToSpace',
            'SpaceToDepth',
            'Split',
            'Concat',
            'TopK'
        ]);
        const name = (value) => value && typeof value.name === 'string' ? value.name.split('\n').shift() : '';
        const values = (arguments_) => {
            const result = [];
            for (const argument of arguments_ || []) {
                for (const value of argument.value || []) {
                    if (value && !value.initializer && name(value)) {
                        result.push(value);
                    }
                }
            }
            return result;
        };
        const type = (node) => {
            if (!node || !node.type) {
                return '';
            }
            return typeof node.type === 'string' ? node.type : node.type.name;
        };
        const candidates = nodes.filter((node) => passthrough.has(type(node))).map((node) => {
            const nodeType = type(node);
            const inputs = values(node.inputs);
            const outputs = values(node.outputs);
            return {
                inputs: nodeType === 'Concat' ? inputs : inputs.slice(0, 1),
                // TopK values preserve the input precision, while indices are int64.
                outputs: nodeType === 'TopK' ? outputs.slice(0, 1) : outputs
            };
        });
        const parent = new Map();
        const find = (value) => {
            const tensorName = name(value);
            if (!parent.has(tensorName)) {
                parent.set(tensorName, tensorName);
            }
            let root = tensorName;
            while (parent.get(root) !== root) {
                root = parent.get(root);
            }
            parent.set(tensorName, root);
            return root;
        };
        const union = (first, second) => {
            const left = find(first);
            const right = find(second);
            if (left !== right) {
                parent.set(right, left);
            }
        };
        const tensors = new Map();
        for (const candidate of candidates) {
            const connected = [...candidate.inputs, ...candidate.outputs];
            for (const value of connected) {
                tensors.set(name(value), value);
                find(value);
            }
            for (let i = 1; i < connected.length; i++) {
                union(connected[0], connected[i]);
            }
        }
        const components = new Map();
        for (const value of tensors.values()) {
            const root = find(value);
            if (!components.has(root)) {
                components.set(root, []);
            }
            components.get(root).push(value);
        }
        for (const component of components.values()) {
            const explicit = component.map((value) => this.get(name(value)))
                .filter((entry) => entry && entry.category === 'activation');
            const labels = new Set(explicit.map((entry) => entry.label));
            if (explicit.length === 0 || labels.size !== 1) {
                continue;
            }
            const source = explicit[0];
            for (const value of component) {
                const tensorName = name(value);
                if (!this.get(tensorName)) {
                    this._inferred.set(tensorName, {
                        name: tensorName,
                        dataType: source.dataType,
                        bitWidth: source.bitWidth,
                        category: 'activation',
                        label: source.label,
                        inferred: true,
                        inferredFrom: source.name,
                        granularity: 'bit-width only',
                        axis: null,
                        blockSize: null,
                        symmetric: null,
                        scale: null,
                        zeroPoint: null,
                        zeroPointLabel: 'zero point',
                        min: null,
                        max: null,
                        rangeSource: null,
                        matched: false
                    });
                }
            }
        }
    }

    _openList(data, category, convert) {
        if (data === undefined || data === null) {
            return;
        }
        if (!Array.isArray(data)) {
            this._parseIssue(`Expected ${category}_encodings to be a list.`);
            return;
        }
        for (const item of data) {
            this._add(convert(item), category);
        }
    }

    _openLegacy(data, category) {
        if (data === undefined || data === null) {
            return;
        }
        if (!data || typeof data !== 'object' || Array.isArray(data)) {
            this._parseIssue(`Expected ${category}_encodings to be an object.`);
            return;
        }
        for (const [name, encodings] of Object.entries(data)) {
            if (!Array.isArray(encodings) || encodings.length === 0) {
                this._parseIssue(`Encoding '${name}' must contain at least one entry.`);
                continue;
            }
            const first = encodings[0];
            const bitWidth = Number(first.bitwidth);
            const dataType = `${String(first.dtype || 'int').toLowerCase()}${Number.isInteger(bitWidth) ? bitWidth : ''}`;
            const entry = {
                name,
                dataType,
                bitWidth: Number.isInteger(bitWidth) ? bitWidth : null,
                scale: encodings.map((encoding) => encoding.scale).filter((value) => value !== undefined),
                zeroPoint: encodings.map((encoding) => encoding.offset).filter((value) => value !== undefined),
                zeroPointLabel: 'offset',
                min: encodings.map((encoding) => encoding.min).filter((value) => value !== undefined),
                max: encodings.map((encoding) => encoding.max).filter((value) => value !== undefined),
                rangeConvention: 'offset',
                axis: null,
                blockSize: null,
                symmetric: aimet.Utility.boolean(first.is_symmetric),
                granularity: encodings.length > 1 ? 'per-channel' : 'per-tensor'
            };
            this._add(entry, category);
        }
    }

    _version1(item) {
        if (!item || typeof item !== 'object') {
            this._parseIssue('Encoding entry must be an object.');
            return null;
        }
        const bitWidth = Number(item.bw);
        const dataType = `${String(item.dtype || 'INT').toLowerCase()}${Number.isInteger(bitWidth) ? bitWidth : ''}`;
        const type = String(item.enc_type || '').toUpperCase();
        let granularity = 'per-tensor';
        if (type === 'PER_CHANNEL') {
            granularity = 'per-channel';
        } else if (type === 'PER_BLOCK') {
            granularity = 'per-block';
        } else if (type === 'LPBQ') {
            granularity = 'lpbq';
        }
        return {
            name: item.name,
            dataType,
            bitWidth: Number.isInteger(bitWidth) ? bitWidth : null,
            scale: item.scale,
            zeroPoint: item.offset,
            zeroPointLabel: 'offset',
            min: item.min === undefined ? null : item.min,
            max: item.max === undefined ? null : item.max,
            rangeConvention: 'offset',
            axis: Number.isInteger(item.axis) ? item.axis : null,
            blockSize: Number.isInteger(item.block_size) ? item.block_size : null,
            symmetric: aimet.Utility.boolean(item.is_sym),
            granularity
        };
    }

    _version2(item) {
        if (!item || typeof item !== 'object') {
            this._parseIssue('Encoding entry must be an object.');
            return null;
        }
        const dataType = String(item.output_dtype || '');
        const match = dataType.match(/(\d+)/);
        const bitWidth = match ? Number(match[1]) : null;
        const blockSize = Number.isInteger(item.block_size) ? item.block_size : null;
        let scale = item.y_scale;
        if (!('y_scale' in item)) {
            scale = {
                perBlock: item.per_block_int_scale,
                perChannel: item.per_channel_float_scale
            };
        }
        let granularity = 'per-tensor';
        if ('per_block_int_scale' in item) {
            granularity = 'lpbq';
        } else if (blockSize === null) {
            granularity = Array.isArray(item.y_scale) ? 'per-channel' : 'per-tensor';
        } else {
            granularity = 'per-block';
        }
        let minimum = item.min === undefined ? null : item.min;
        let maximum = item.max === undefined ? null : item.max;
        if (item.y_min !== undefined) {
            minimum = item.y_min;
        }
        if (item.y_max !== undefined) {
            maximum = item.y_max;
        }
        return {
            name: item.name,
            dataType,
            bitWidth,
            scale,
            zeroPoint: item.y_zero_point === undefined ? 0 : item.y_zero_point,
            zeroPointLabel: 'zero point',
            min: minimum,
            max: maximum,
            rangeConvention: 'zero-point',
            axis: Number.isInteger(item.axis) ? item.axis : null,
            blockSize,
            symmetric: null,
            granularity
        };
    }

    _add(entry, category) {
        if (!entry) {
            return;
        }
        if (typeof entry.name !== 'string' || !entry.name) {
            this._parseIssue('Encoding entry is missing a tensor name.');
            return;
        }
        if (!entry.dataType) {
            this._parseIssue(`Encoding '${entry.name}' is missing a data type.`);
            return;
        }
        if (this._entries.has(entry.name)) {
            this._parseIssue(`Duplicate encoding for tensor '${entry.name}'.`);
            return;
        }
        const range = aimet.Utility.quantizationRange(entry);
        entry.min = range.min;
        entry.max = range.max;
        entry.rangeSource = range.source;
        delete entry.rangeConvention;
        entry.category = category;
        entry.label = `${category === 'parameter' ? 'W' : 'A'}${entry.bitWidth || entry.dataType.toUpperCase()}`;
        entry.matched = false;
        this._entries.set(entry.name, entry);
    }

    _parseIssue(message) {
        this._issues.push({ severity: 'error', source: 'parse', message });
    }
};

aimet.Utility = class {

    static quantizationRange(entry) {
        const hasValue = (value) => value !== undefined && value !== null && (!Array.isArray(value) || value.length > 0);
        if (hasValue(entry.min) || hasValue(entry.max)) {
            return { min: entry.min, max: entry.max, source: 'explicit' };
        }
        const bitWidth = Number(entry.bitWidth);
        if (!Number.isInteger(bitWidth) || bitWidth < 1 || bitWidth > 32) {
            return { min: null, max: null, source: null };
        }
        let qmin = 0;
        let qmax = (2 ** bitWidth) - 1;
        let sign = 1;
        if (entry.rangeConvention === 'zero-point') {
            const match = String(entry.dataType || '').toLowerCase().match(/^(u?)int(\d+)$/);
            if (!match || Number(match[2]) !== bitWidth) {
                return { min: null, max: null, source: null };
            }
            if (match[1] !== 'u') {
                qmin = -(2 ** (bitWidth - 1));
                qmax = (2 ** (bitWidth - 1)) - 1;
            }
            sign = -1;
        } else if (entry.rangeConvention !== 'offset') {
            return { min: null, max: null, source: null };
        }
        const flatten = (value, values = []) => {
            if (Array.isArray(value)) {
                for (const item of value) {
                    flatten(item, values);
                }
            } else if (Number.isFinite(Number(value))) {
                values.push(Number(value));
            }
            return values;
        };
        const scales = flatten(entry.scale);
        const points = flatten(entry.zeroPoint);
        if (scales.length === 0 || points.length === 0 || scales.some((value) => value <= 0)) {
            return { min: null, max: null, source: null };
        }
        const length = Math.max(scales.length, points.length);
        if (scales.length !== 1 && scales.length !== length || points.length !== 1 && points.length !== length) {
            return { min: null, max: null, source: null };
        }
        const normalize = (value) => Number(value.toPrecision(12));
        const minimum = [];
        const maximum = [];
        for (let index = 0; index < length; index++) {
            const scale = scales[scales.length === 1 ? 0 : index];
            const point = points[points.length === 1 ? 0 : index];
            minimum.push(normalize((qmin + (sign * point)) * scale));
            maximum.push(normalize((qmax + (sign * point)) * scale));
        }
        return {
            min: length === 1 ? minimum[0] : minimum,
            max: length === 1 ? maximum[0] : maximum,
            source: 'derived'
        };
    }

    static boolean(value) {
        if (typeof value === 'boolean') {
            return value;
        }
        if (typeof value === 'string') {
            const text = value.toLowerCase();
            if (text === 'true') {
                return true;
            }
            if (text === 'false') {
                return false;
            }
        }
        return null;
    }

    static count(value) {
        if (!Array.isArray(value)) {
            return value === undefined || value === null ? 0 : 1;
        }
        return value.reduce((sum, item) => sum + aimet.Utility.count(item), 0);
    }

    static format(value) {
        if (value === undefined || value === null || Array.isArray(value) && value.length === 0) {
            return '';
        }
        if (!Array.isArray(value) && typeof value !== 'object') {
            return String(value);
        }
        const values = [];
        const flatten = (item) => {
            if (Array.isArray(item)) {
                for (const value of item) {
                    flatten(value);
                }
            } else if (item && typeof item === 'object') {
                for (const [name, value] of Object.entries(item)) {
                    values.push(`${name}: ${aimet.Utility.format(value)}`);
                }
            } else if (item !== undefined) {
                values.push(String(item));
            }
        };
        flatten(value);
        const preview = values.slice(0, 4).join(', ');
        return values.length > 4 ? `[${preview}, …] (${values.length})` : `[${preview}]`;
    }
};

export const EncodingFile = aimet.EncodingFile;
export const Utility = aimet.Utility;
