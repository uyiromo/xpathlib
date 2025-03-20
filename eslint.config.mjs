import stylisticJs from '/usr/lib/node_modules/@stylistic/eslint-plugin-js/dist/index.js'

export default [
    {
        plugins: {
            '@stylistic/js': stylisticJs
        },
        files: ["**/*.js"],
        rules: {
            '@stylistic/js/array-bracket-newline': ['error', 'always'],
            '@stylistic/js/array-element-newline': ['error', 'always'],
            '@stylistic/js/eol-last': ['error', 'always'],
            '@stylistic/js/comma-spacing': ["error", { "before": false, "after": true }],
            '@stylistic/js/indent': ['error', 4],
            '@stylistic/js/semi': ['error', "always"]
        }
    }
];
