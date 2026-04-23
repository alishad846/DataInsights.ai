import path from 'path';

/**
 * Sanitizes a filename:
 * 1. Only allows a-zA-Z0-9._-
 * 2. Normalizes multiple dots to a single dot.
 * 3. Trims the name part to a maximum of 30 characters.
 * 4. Ensures the extension is lowercase.
 */
export const sanitizeFilename = (filename) => {
    if (!filename) return 'unnamed_dataset';
    
    const ext = path.extname(filename);
    let name = path.basename(filename, ext);
    
    // 1. Sanitize characters: allow a-zA-Z0-9._- and replace others with underscore
    name = name.replace(/[^a-zA-Z0-9._-]/g, '_');
    
    // 2. Normalize multiple dots and underscores
    name = name.replace(/\.+/g, '.');
    name = name.replace(/_+/g, '_');
    
    // 3. Trim name part to max 30 characters
    if (name.length > 30) {
        name = name.substring(0, 30);
    }
    
    // Clean up trailing dots or underscores from name part before adding extension
    name = name.replace(/[._]+$/, '');
    
    const finalExt = ext.toLowerCase();
    
    // Final check for empty name (if everything was sanitized away)
    if (!name) name = 'dataset';
    
    return `${name}${finalExt}`;
};
