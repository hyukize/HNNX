
export const classify = (items, name, accept) => {
    items = Array.from(items || []);
    const isEncoding = (item) => {
        const value = name(item).toLowerCase();
        return value.endsWith('.encodings') || value.endsWith('.encodings.json');
    };
    const attachments = items.filter((item) => isEncoding(item));
    const candidates = items.filter((item) => !isEncoding(item) && accept(item));
    const onnx = candidates.filter((item) => name(item).toLowerCase().endsWith('.onnx'));
    // An ONNX external-data sidecar may itself match a generic loader. When an
    // explicit ONNX file exists, only ONNX files are primary model candidates.
    const models = onnx.length > 0 ? onnx : candidates;
    return { attachments, models };
};
