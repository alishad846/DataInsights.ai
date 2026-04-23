export default function Table({ columns, data }) {
    return (
        <table border="1" width="100%">
            <thead>
                <tr>
                    {columns.map((col, i) => (
                        <th key={i}>{col}</th>
                    ))}
                </tr>
            </thead>

            <tbody>
                {data.map((row, i) => (
                    <tr key={i}>
                        {Object.values(row).map((val, j) => (
                            <td key={j}>{val}</td>
                        ))}
                    </tr>
                ))}
            </tbody>
        </table>
    );
}
