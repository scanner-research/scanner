var path = require('path');
var HtmlWebpackPlugin = require('html-webpack-plugin');


const common = {
  // Important! Do not remove ''. If you do, imports without
  // an extension won't work anymore!
  resolve: {
    extensions: ['', '.js', '.jsx']
  }
}

module.exports = {
    entry: "./src/js/index.jsx",
    module: {
        loaders: [
            {
                test: /src\/.+.jsx?$/,
                include: /src/,
                exclude: /node_modules/,
                // Enable caching for improved performance during development
                // It uses default OS directory by default. If you need
                // something more custom, pass a path to it.
                // I.e., babel?cacheDirectory=<path>
                loaders: ['babel?cacheDirectory'],
                // Parse only app files! Without this it will go through
                // the entire project. In addition to being slow,
                // that will most likely result in an error.
            },
            { test: /\.scss$/, loaders: ['style', 'css', 'sass'] },
            { test: /\.css$/, loader: "style!css" },
            { test: /\.html$/, loader: 'html' },
        ]
    },
    output: {
        path: path.join(__dirname, 'dist'),
        filename: "bundle.js"
    },
    plugins: [
        new HtmlWebpackPlugin({
            template: 'src/index.html'
        })
    ]
};
